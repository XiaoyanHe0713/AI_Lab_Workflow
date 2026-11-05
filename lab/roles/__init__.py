from .base import (  # noqa: F401
    FRONTIER_KINDS,
    KINDS,
    MODES,
    AnthropicProvider,
    BudgetedProvider,
    Completion,
    OpenAIChatProvider,
    Provider,
    RecordingProvider,
    ReplayProvider,
    Role,
    RoleResult,
    conversation_opening,
    extract_code,
    make_provider,
    make_role,
    request_key,
    strip_thinking,
)

OpenAICompatibleProvider = OpenAIChatProvider  # older name
