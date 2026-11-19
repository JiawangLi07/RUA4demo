"""Model decisions inside an owned SDK session. Hardware IO stays on the station thread."""
import ast
import json
import math
from itertools import count

from pyrualean._backend import Ledger
from pyrualean.arms import ArmToolkit
from robot import SharpaRobot
from task_library import atomic_json


class SessionRobot(SharpaRobot):
    def grasp(self, plan):
        return self._readonly("grasp", {"plan":plan})

    def move_relative(self, xyz_m):
        return self._stateful("cartesian", {"values":xyz_m,"relative":True}, build=dict)

    def move_pose(self, xyz_xyzw):
        return self._stateful("cartesian", {"values":xyz_xyzw,"relative":False}, build=dict)

    def measure(self, bbox):
        return self._readonly("measure", {"bbox":bbox})

    def arm_step(self, joint, delta_deg):
        return self._stateful("arm_step", {"joint":joint,"delta_deg":delta_deg}, build=dict)

    def step(self, joint, delta_deg):
        return self._stateful("step", {"joint":joint,"delta_deg":delta_deg}, build=dict)


def decode_step(code, names):
    expr = ast.parse(code,mode="eval").body
    if (isinstance(expr,ast.Call) and isinstance(expr.func,ast.Attribute)
            and isinstance(expr.func.value,ast.Name) and expr.func.value.id=="robo"
            and expr.func.attr=="grasp" and len(expr.args)==1 and not expr.keywords):
        from .grasp_program import Program
        plan=ast.literal_eval(expr.args[0])
        Program.parse(plan,[plan.get("grasp")])
        return
    if (isinstance(expr,ast.Call) and isinstance(expr.func,ast.Attribute)
            and isinstance(expr.func.value,ast.Name) and expr.func.value.id=="robo"
            and expr.func.attr in ("measure","move_relative","move_pose") and len(expr.args)==1 and not expr.keywords):
        box=ast.literal_eval(expr.args[0])
        if not isinstance(box,list) or len(box) not in ({"measure":(4,), "move_relative":(3,), "move_pose":(3,7)}[expr.func.attr]) or not all(type(v) in (int,float) and math.isfinite(v) for v in box):
            raise ValueError("Expected four finite pixel coordinates")
        return
    if not (isinstance(expr,ast.Call) and isinstance(expr.func,ast.Attribute)
            and isinstance(expr.func.value,ast.Name) and expr.func.value.id=="robo"
            and expr.func.attr in ("step","arm_step") and len(expr.args)==2 and not expr.keywords):
        raise ValueError("Expected robo.step(joint_name, delta_deg)")
    joint, delta = [ast.literal_eval(arg) for arg in expr.args]
    valid = names if expr.func.attr=="step" else [f"Joint{i}_R" for i in range(1,8)]
    if joint not in valid or type(delta) not in (float,int) or not math.isfinite(delta) or not 0 < abs(delta) or (expr.func.attr=="step" and abs(delta)>2):
        raise ValueError("Use a listed hand joint and a finite nonzero relative step; hand steps are at most 2 degrees")


def validate_session_cell(code,names):
    tree=ast.parse(code)
    if not 1<=len(tree.body)<=12:
        raise ValueError("Expected up to twelve sequential measurement/action statements")
    statements=[]
    def collect(body):
        for stmt in body:
            if isinstance(stmt,ast.If):
                for node in ast.walk(stmt.test):
                    if not isinstance(node,(ast.Subscript,ast.Name,ast.Constant,ast.Load)):
                        raise ValueError("Conditions must read returned measurement keys")
                collect(stmt.body)
                collect(stmt.orelse)
            else:
                statements.append(stmt)
    collect(tree.body)
    if len(statements)>12:
        raise ValueError("Too many statements")
    literals={}
    for stmt in statements:
        if isinstance(stmt,ast.Assign):
            if len(stmt.targets)!=1 or not isinstance(stmt.targets[0],ast.Name) or stmt.targets[0].id in ("robo","print"):
                raise ValueError("Invalid assignment")
            value=stmt.value
            try:
                literals[stmt.targets[0].id]=ast.literal_eval(value)
                continue
            except (ValueError,TypeError):
                literals.pop(stmt.targets[0].id,None)
        elif isinstance(stmt,ast.Expr):
            value=stmt.value
        else:
            raise ValueError("Only measurement, action and print statements")
        class LiteralArguments(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id in literals:
                    return ast.copy_location(ast.parse(repr(literals[node.id]),mode="eval").body,node)
                return node
        value=LiteralArguments().visit(value)
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=="print":
            for arg in value.args:
                if isinstance(arg,ast.Call):
                    decode_step(ast.unparse(arg),names)
                elif not isinstance(arg,(ast.Name,ast.Constant)):
                    raise ValueError("Invalid print argument")
        else:
            decode_step(ast.unparse(value),names)



class SessionDialogue:
    def __init__(self, channel, out, profile):
        self.channel, self.out, self.profile = channel, out, profile
        self.history = []
        self.commands = 0
        self.results = []
        self.proposal = None

    def env_steps(self):
        return self.commands

    def solved(self):
        return False

    def execute_tool(self, name, inputs):
        if name=="grasp":
            from .free_grasp import requested_program
            from .station import Station
            try:
                station=Station.read(self.profile_path)
                program=requested_program(inputs["plan"],station)
            except (ValueError,KeyError) as exc:
                self.last_result={"accepted":False,"reason":str(exc)}
            else:
                self.proposal=program
                self.last_result={"accepted":True,"next":"verified contact trial"}
                atomic_json(self.out/"grasp-request.json",inputs["plan"])
            self.results.append({"tool":name,"inputs":inputs,"result":self.last_result})
            return self.last_result
        if name=="measure":
            self.last_result=self.measure(inputs["bbox"])
            self.results.append({"tool":name,"inputs":inputs,"result":self.last_result})
            return self.last_result
        if name=="cartesian":
            self.last_result=self.cartesian_move(inputs["values"],inputs["relative"])
            self.results.append({"tool":name,"inputs":inputs,"result":self.last_result})
            self.commands+=1
            return self.last_result
        if name not in ("step","arm_step"):
            raise ValueError("Unknown session primitive")
        self.last_result = (self.move if name=="step" else self.arm_move)(inputs["joint"], inputs["delta_deg"])
        self.results.append({"tool":name,"inputs":inputs,"result":self.last_result})
        self.commands += 1
        return self.last_result

    def run(self, move, monitor, snapshot, names, first, arm_move=None, measure=None, cartesian_move=None):
        self.cartesian_move=cartesian_move
        self.measure=measure
        self.arm_move=arm_move
        self.move=move
        robot=SessionRobot(self,ledger=Ledger())
        cells=ArmToolkit(robot,arm="cells",workspace=self.out/"session-cells",images=False,feedback="pure",max_turns=None)
        for turn in count():
            observed=snapshot()
            request={"turn":f"hold-{turn}", "session_active":True, "task":"右手抓取桌上的羽毛球",
                     "profile":self.profile,"joint_names":names,"observation":observed["feedback"],
                     "image_base64":observed["image_base64"],"first_step":first,"history":self.history,
                     "last_turn":False}
            reply=self.channel.request("diagnostic_request",request,watch=monitor)
            atomic_json(self.out/f"session-decision-{turn}.json",reply)
            if reply["action"]=="stop":
                return reply["reason"]
            if reply["action"]!="cell":
                raise ValueError("Invalid unloaded-session decision")
            validate_session_cell(reply["source"],names)
            self.last_result=None
            self.results=[]
            result=cells.execute_tool("python",{"code":reply["source"]}).result
            robot._ledger.write_jsonl(self.out/"session-calls.jsonl")
            if result.get("error") or result.get("exception") or self.last_result is None:
                raise RuntimeError(result.get("error") or result.get("exception") or "Missing hardware result")
            self.history.append({"code":reply["source"],"reason":reply["reason"],"feedback":self.last_result,"tool_results":self.results,"stdout":result.get("stdout","")})
            atomic_json(self.out/"session-history.json",self.history)
            if self.proposal is not None:
                return "grasp_proposal_ready"
