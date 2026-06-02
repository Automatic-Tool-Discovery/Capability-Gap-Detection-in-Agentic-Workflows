from src.schemas import AgentTrace, Prediction
from src.taxonomy import FailureType


MISSING_TOOL_KEYWORDS = {
    "pdf": "pdf_table_extractor",
    "ocr": "ocr_image_reader",
    "image": "ocr_image_reader",
    "email": "email_sender",
    "audio": "audio_transcriber",
    "transcribe": "audio_transcriber",
}


def classify_trace(trace: AgentTrace) -> Prediction:
    task_lower = trace.user_task.lower()
    errors = " ".join(
        call.error or "" for call in trace.tool_calls
    ).lower()
    failure_explanation = (trace.failure_explanation or "").lower()

    evidence = []

    # F6: obvious missing capability cases
    for keyword, missing_tool in MISSING_TOOL_KEYWORDS.items():
        if keyword in task_lower and missing_tool not in trace.available_tools:
            evidence.append(
                f"Task mentions '{keyword}', but required tool '{missing_tool}' is not available."
            )
            return Prediction(
                trace_id=trace.trace_id,
                predicted_label=FailureType.MISSING_CAPABILITY_GAP.value,
                confidence=0.75,
                evidence=evidence,
                new_tool_needed=True,
            )
        
    # F1: reasoning/planning error when calculator is used directly on raw structured data
    if "csv" in task_lower and any(call.tool_name == "calculator" for call in trace.tool_calls):
        if "csv_reader" in trace.available_tools:
            evidence.append(
                "The task involved CSV data and the csv_reader tool was available, "
                "but the agent used the calculator directly on raw CSV content."
            )
            return Prediction(
                trace_id=trace.trace_id,
                predicted_label=FailureType.REASONING_OR_PLANNING_ERROR.value,
                confidence=0.7,
                evidence=evidence,
                new_tool_needed=False,
        )

    # F5: documentation/schema error
    if (
        "documentation" in failure_explanation
        or "tool description" in failure_explanation
        or "described" in failure_explanation
        or "misleading" in failure_explanation
        or "schema required" in failure_explanation
    ):
        evidence.append(
            "The failure explanation indicates the tool documentation or schema was misleading."
        )
        return Prediction(
            trace_id=trace.trace_id,
            predicted_label=FailureType.TOOL_DOCUMENTATION_OR_SCHEMA_ERROR.value,
            confidence=0.7,
            evidence=evidence,
            new_tool_needed=False,
        )
    
    # F8: environment or state error
    if (
        "filenotfounderror" in errors
        or "does not exist" in errors
        or "not logged in" in errors
        or "authenticationerror" in errors
        or "not authenticated" in errors
        or "missing from the environment" in failure_explanation
        or "environment state" in failure_explanation
    ):
        evidence.append(
            "The correct tool exists, but the environment or state prevents successful execution."
        )
        return Prediction(
            trace_id=trace.trace_id,
            predicted_label=FailureType.ENVIRONMENT_OR_STATE_ERROR.value,
            confidence=0.75,
            evidence=evidence,
            new_tool_needed=False,
        )
    
    # F3: wrong parameters / schema issue
    if "invalid" in errors or "expected" in errors or "missing argument" in errors:
        evidence.append("Tool returned an argument or schema-related error.")
        return Prediction(
            trace_id=trace.trace_id,
            predicted_label=FailureType.WRONG_TOOL_PARAMETERS.value,
            confidence=0.65,
            evidence=evidence,
            new_tool_needed=False,
        )

    # F4: runtime error
    if "timeout" in errors or "network" in errors or "rate limit" in errors:
        evidence.append("Tool failed due to runtime or external execution issue.")
        return Prediction(
            trace_id=trace.trace_id,
            predicted_label=FailureType.TOOL_RUNTIME_ERROR.value,
            confidence=0.65,
            evidence=evidence,
            new_tool_needed=False,
        )

    # Default: reasoning/planning error
    evidence.append("Tools may be available, but the trace suggests incorrect use or planning.")
    return Prediction(
        trace_id=trace.trace_id,
        predicted_label=FailureType.REASONING_OR_PLANNING_ERROR.value,
        confidence=0.5,
        evidence=evidence,
        new_tool_needed=False,
    )