# graph/pipeline.py

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from models.schemas import (
    ComplianceState
)


class CompliancePipeline:

    def __init__(
        self,
        compliance_agent,
        risk_agent,
        rewrite_agent,
        validator_agent
    ):

        self.compliance_agent = (
            compliance_agent
        )

        self.risk_agent = (
            risk_agent
        )

        self.rewrite_agent = (
            rewrite_agent
        )

        self.validator_agent = (
            validator_agent
        )

    # =====================================
    # ROUTING FUNCTIONS
    # =====================================

    @staticmethod
    def compliance_router(
        state: ComplianceState
    ):

        if (
            state[
                "compliance_verdict"
            ] == "compliant"
        ):

            return "end"

        return "risk"

    @staticmethod
    def validator_router(
        state: ComplianceState
    ):

        if state[
            "manual_review_flag"
        ]:

            return "end"

        if state[
            "validation_passed"
        ]:

            return "end"

        return "rewrite"

    # =====================================
    # BUILD GRAPH
    # =====================================

    def build(self):

        graph = StateGraph(
            ComplianceState
        )

        # ==========================
        # Nodes
        # ==========================

        graph.add_node(
            "compliance_agent",
            self.compliance_agent.run
        )

        graph.add_node(
            "risk_agent",
            self.risk_agent.run
        )

        graph.add_node(
            "rewrite_agent",
            self.rewrite_agent.run
        )

        graph.add_node(
            "validator_agent",
            self.validator_agent.run
        )

        # ==========================
        # Start
        # ==========================

        graph.add_edge(
            START,
            "compliance_agent"
        )

        # ==========================
        # Compliance Routing
        # ==========================

        graph.add_conditional_edges(
            "compliance_agent",
            self.compliance_router,
            {
                "risk":
                    "risk_agent",

                "end":
                    END
            }
        )

        # ==========================
        # Standard Flow
        # ==========================

        graph.add_edge(
            "risk_agent",
            "rewrite_agent"
        )

        graph.add_edge(
            "rewrite_agent",
            "validator_agent"
        )

        # ==========================
        # Validator Routing
        # ==========================

        graph.add_conditional_edges(
            "validator_agent",
            self.validator_router,
            {
                "rewrite":
                    "rewrite_agent",

                "end":
                    END
            }
        )

        return graph.compile()