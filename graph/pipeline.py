# graph/pipeline.py

from langgraph.graph import StateGraph, START, END
from models.schemas import ComplianceState


class CompliancePipeline:

    def __init__(
        self,
        compliance_agent,
        rewrite_agent,
        validator_agent
    ):
        self.compliance_agent = compliance_agent
        self.rewrite_agent    = rewrite_agent
        self.validator_agent  = validator_agent

    # =====================================
    # ROUTING
    # =====================================

    @staticmethod
    def compliance_router(state: ComplianceState):
        # Fix 4 — hard stop for compliant clauses, no downstream calls
        if state["compliance_verdict"] == "compliant":
            return END
        return "rewrite_agent"

    @staticmethod
    def validator_router(state: ComplianceState):
        if state["manual_review_flag"]:
            return END
        if state["validation_passed"]:
            return END
        return "rewrite_agent"

    # =====================================
    # BUILD GRAPH
    # =====================================

    def build(self):
        graph = StateGraph(ComplianceState)

        graph.add_node("compliance_agent", self.compliance_agent.run)
        graph.add_node("rewrite_agent",    self.rewrite_agent.run)
        graph.add_node("validator_agent",  self.validator_agent.run)

        graph.add_edge(START, "compliance_agent")

        graph.add_conditional_edges(
            "compliance_agent",
            self.compliance_router,
            {"rewrite_agent": "rewrite_agent", END: END}
        )

        graph.add_edge("rewrite_agent", "validator_agent")

        graph.add_conditional_edges(
            "validator_agent",
            self.validator_router,
            {"rewrite_agent": "rewrite_agent", END: END}
        )

        return graph.compile()
