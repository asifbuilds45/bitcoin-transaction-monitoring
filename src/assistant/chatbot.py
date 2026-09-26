"""
Offline AI Investigation Assistant Engine
SIH Problem Statement 26146: AI-Powered Monitoring & Analysis of Bitcoin Transaction Traffic

Coordinates conversation state, context memory (active TXID, active Case),
targeted evidence retrieval, and offline LLM/deterministic generation.
Provides transparent auditability with an exact 'Evidence Used' breakdown.
"""

from typing import Dict, Any, List, Optional
from src.assistant.investigation_knowledge import InvestigationKnowledgeStore
from src.assistant.evidence_retrieval import EvidenceRetriever, RetrievedEvidence
from src.assistant.offline_llm import OfflineLLMManager


class InvestigationChatbot:
    """
    Forensic chatbot managing conversational state and grounding answers strictly in retrieved evidence.
    """

    def __init__(self, knowledge_store: Any):
        if isinstance(knowledge_store, InvestigationKnowledgeStore):
            self.store = knowledge_store
        elif hasattr(knowledge_store, "empty"):  # pandas DataFrame
            self.store = InvestigationKnowledgeStore(context={"scored_df": knowledge_store})
        elif isinstance(knowledge_store, dict):
            self.store = InvestigationKnowledgeStore(context=knowledge_store)
        else:
            self.store = InvestigationKnowledgeStore()
        self.retriever = EvidenceRetriever(self.store)
        self.llm_manager = OfflineLLMManager()
        self.active_context: Dict[str, Any] = {
            "active_txid": None,
            "active_case_id": None,
            "active_wallet": None
        }

    def reset_context(self) -> None:
        """Reset conversation context when new investigation data is loaded."""
        self.active_context = {
            "active_txid": None,
            "active_case_id": None,
            "active_wallet": None
        }

    def set_active_entity(self, entity_id: str, entity_type: str = "TXID") -> None:
        """Explicitly focus the assistant on a specific entity."""
        if entity_type.upper() == "TXID":
            self.active_context["active_txid"] = entity_id.upper()
        elif entity_type.upper() in ["CASE", "CASE_ID"]:
            self.active_context["active_case_id"] = entity_id.upper()
        elif entity_type.upper() == "WALLET":
            self.active_context["active_wallet"] = entity_id.upper()

    def ask(self, query: str, selected_txid: Optional[str] = None) -> Dict[str, Any]:
        """
        Process an investigator query.
        
        Returns:
            Dict containing response, evidence_used, intent, active_context, etc.
        """
        if selected_txid:
            self.set_active_entity(selected_txid, "TXID")

        # 1. Retrieve evidence using query and active context
        evidence: RetrievedEvidence = self.retriever.retrieve(query, context=self.active_context)

        # 2. Update context tracking if a specific target was identified
        if evidence.target_entity and evidence.target_type:
            if evidence.target_type == "TXID":
                self.active_context["active_txid"] = evidence.target_entity
                # If transaction has an assigned case, set it too
                if "case_id" in evidence.structured_data and evidence.structured_data["case_id"] != "N/A (Isolated)":
                    self.active_context["active_case_id"] = evidence.structured_data["case_id"]
            elif evidence.target_type == "CASE":
                self.active_context["active_case_id"] = evidence.target_entity
            elif evidence.target_type == "WALLET":
                self.active_context["active_wallet"] = evidence.target_entity

        # 3. Generate synthesized response
        response_text = self.llm_manager.generate_response(query, evidence)

        # 4. Structure auditable evidence items
        structured_evidence = []
        if evidence.is_supported:
            for item in evidence.evidence_items:
                if isinstance(item, dict):
                    structured_evidence.append(item)
                elif ":" in str(item):
                    parts = str(item).split(":", 1)
                    structured_evidence.append({
                        "source_tag": parts[0].strip(),
                        "fact": parts[1].strip()
                    })
                else:
                    structured_evidence.append({
                        "source_tag": evidence.target_type or "RECORD",
                        "fact": str(item)
                    })

        return {
            "response": response_text,
            "evidence_used": structured_evidence,
            "intent": evidence.intent,
            "target_entity": evidence.target_entity,
            "target_type": evidence.target_type,
            "active_context": self.active_context.copy(),
            "is_supported": evidence.is_supported,
            "unsupported_reason": evidence.unsupported_reason,
            "model_status": self.llm_manager.get_status()
        }
