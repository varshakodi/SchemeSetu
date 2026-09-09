"""HTTP 429 means two different things; the response must differ.

A per-minute meter clears by waiting. A daily one does not, and retrying it
stalls the caller for minutes before a fallback can engage. This classifier
has been wrong in both directions:

  1. It read the WAIT LENGTH -- treating retry-after > 30s as terminal. Groq
     reports its per-minute limit with "try again in 51.84s" as well, so real
     throttles were fatal and eval runs died.
  2. Fixing that, it matched "billing" and bare "quota". Every Groq 429 ends
     with "Upgrade to Dev Tier today at .../settings/billing", so every
     throttle became terminal again -- and that run stopped at question 1.

The bodies below are real, captured from live 429s during eval runs.
"""

from agent.llm import is_daily_quota

GROQ_SUFFIX = (" Need more tokens? Upgrade to Dev Tier today at "
               "https://console.groq.com/settings/billing")

GROQ_PER_MINUTE = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` "
    "service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 7500, "
    "Requested 900. Please try again in 3.5s." + GROQ_SUFFIX
)
GROQ_PER_DAY = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` "
    "service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199602, "
    "Requested 518. Please try again in 51.84s." + GROQ_SUFFIX
)
GEMINI_DAILY = (
    "You exceeded your current quota, please check your plan and billing details. "
    "Quota exceeded for metric: generativelanguage.googleapis.com/"
    "generate_content_free_tier_requests, limit: 20"
)


def test_per_minute_throttle_is_retryable():
    assert not is_daily_quota(GROQ_PER_MINUTE)


def test_per_day_quota_is_terminal():
    assert is_daily_quota(GROQ_PER_DAY)


def test_gemini_daily_quota_is_terminal():
    assert is_daily_quota(GEMINI_DAILY)


def test_billing_link_alone_does_not_make_it_terminal():
    """The regression that stopped a run at question 1."""
    assert not is_daily_quota("Please try again in 2s." + GROQ_SUFFIX)


def test_long_retry_after_alone_does_not_make_it_terminal():
    """The earlier regression: wait length is not the signal."""
    assert not is_daily_quota(
        "on tokens per minute (TPM): Limit 8000. Please try again in 58.2s.")
