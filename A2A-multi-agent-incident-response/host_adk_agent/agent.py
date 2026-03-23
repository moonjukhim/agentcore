import httpx
import json
import logging
import os
import uuid
from urllib.parse import quote

from bedrock_agentcore.identity.auth import requires_access_token
from strands import Agent, tool
from strands.models import BedrockModel
from prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

IS_DOCKER = os.getenv("DOCKER_CONTAINER", "0") == "1"
MODEL_ID = os.getenv(
    "STRANDS_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

if IS_DOCKER:
    from utils import get_ssm_parameter, get_aws_info
else:
    from host_adk_agent.utils import get_ssm_parameter, get_aws_info

# AWS and agent configuration
account_id, region = get_aws_info()

MONITOR_AGENT_ID = get_ssm_parameter("/monitoragent/agentcore/runtime-id")
MONITOR_PROVIDER_NAME = get_ssm_parameter("/monitoragent/agentcore/provider-name")
MONITOR_AGENT_ARN = (
    f"arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/{MONITOR_AGENT_ID}"
)

WEBSEARCH_AGENT_ID = get_ssm_parameter("/websearchagent/agentcore/runtime-id")
WEBSEARCH_PROVIDER_NAME = get_ssm_parameter("/websearchagent/agentcore/provider-name")
WEBSEARCH_AGENT_ARN = (
    f"arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/{WEBSEARCH_AGENT_ID}"
)


def _get_a2a_endpoint(agent_arn: str) -> str:
    """Get the A2A JSON-RPC endpoint for an agent runtime."""
    encoded_arn = quote(agent_arn, safe="")
    return (
        f"https://bedrock-agentcore.{region}.amazonaws.com"
        f"/runtimes/{encoded_arn}/invocations"
    )


def _get_agent_card_url(agent_arn: str) -> str:
    """Get the agent card URL for an agent runtime."""
    encoded_arn = quote(agent_arn, safe="")
    return (
        f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/"
        f"{encoded_arn}/invocations/.well-known/agent-card.json"
    )


def _get_auth_token(provider_name: str) -> str:
    """Get OAuth2 bearer token for agent communication."""

    @requires_access_token(
        provider_name=provider_name,
        scopes=[],
        auth_flow="M2M",
        into="bearer_token",
        force_authentication=True,
    )
    def _get_token(bearer_token: str = str()) -> str:
        return bearer_token

    return _get_token()


def _send_a2a_message(
    agent_arn: str,
    provider_name: str,
    query: str,
    session_id: str,
    actor_id: str,
) -> str:
    """Send a message to a remote agent via A2A JSON-RPC 2.0 protocol."""
    endpoint = _get_a2a_endpoint(agent_arn)
    token = _get_auth_token(provider_name)

    # A2A JSON-RPC 2.0 message/send request
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": query}],
                "messageId": str(uuid.uuid4()),
            }
        },
        "id": str(uuid.uuid4()),
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actorid": actor_id,
    }

    with httpx.Client(timeout=httpx.Timeout(timeout=300.0)) as client:
        response = client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()

    # Extract text from A2A JSON-RPC response
    if "result" in result:
        task_result = result["result"]
        # Check artifacts for response text
        if "artifacts" in task_result:
            texts = []
            for artifact in task_result["artifacts"]:
                for part in artifact.get("parts", []):
                    if part.get("kind") == "text":
                        texts.append(part["text"])
            if texts:
                return "\n".join(texts)
        # Fallback to status message
        if "status" in task_result:
            message = task_result["status"].get("message", {})
            texts = []
            for part in message.get("parts", []):
                if part.get("kind") == "text":
                    texts.append(part["text"])
            if texts:
                return "\n".join(texts)

    return json.dumps(result)


def _fetch_agent_card(agent_arn: str, provider_name: str) -> dict:
    """Fetch agent card from A2A well-known endpoint."""
    url = _get_agent_card_url(agent_arn)
    token = _get_auth_token(provider_name)

    with httpx.Client(timeout=30.0) as client:
        response = client.get(url, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        return response.json()


def create_a2a_tools(session_id: str, actor_id: str):
    """Create tool functions for A2A communication with sub-agents."""

    @tool
    def call_monitor_agent(query: str) -> str:
        """Send a monitoring query to the CloudWatch monitoring agent.

        Use this tool for CloudWatch metrics, logs, alarms, and AWS monitoring data.
        Handles EC2/Lambda/RDS metrics, log group queries, error searches,
        and alarm states.

        Args:
            query: The monitoring query to send to the agent.
        """
        logger.info(f"Calling monitor agent with query: {query[:100]}...")
        return _send_a2a_message(
            MONITOR_AGENT_ARN, MONITOR_PROVIDER_NAME, query, session_id, actor_id
        )

    @tool
    def call_websearch_agent(query: str) -> str:
        """Send a query to the web search agent for AWS troubleshooting and documentation.

        Use this tool for finding AWS solutions, documentation, best practices,
        error resolution steps, and architectural guidance.

        Args:
            query: The search query to send to the agent.
        """
        logger.info(f"Calling websearch agent with query: {query[:100]}...")
        return _send_a2a_message(
            WEBSEARCH_AGENT_ARN, WEBSEARCH_PROVIDER_NAME, query, session_id, actor_id
        )

    return [call_monitor_agent, call_websearch_agent]


def fetch_agents_cards() -> dict:
    """Fetch agent cards for all sub-agents."""
    agents_info = {}

    try:
        monitor_card = _fetch_agent_card(MONITOR_AGENT_ARN, MONITOR_PROVIDER_NAME)
        agents_info["monitor_agent"] = {
            "agent_card_url": _get_agent_card_url(MONITOR_AGENT_ARN),
            "agent_card": monitor_card,
        }
    except Exception as e:
        logger.error(f"Failed to fetch monitor agent card: {e}")

    try:
        websearch_card = _fetch_agent_card(
            WEBSEARCH_AGENT_ARN, WEBSEARCH_PROVIDER_NAME
        )
        agents_info["websearch_agent"] = {
            "agent_card_url": _get_agent_card_url(WEBSEARCH_AGENT_ARN),
            "agent_card": websearch_card,
        }
    except Exception as e:
        logger.error(f"Failed to fetch websearch agent card: {e}")

    return agents_info


def create_host_agent(session_id: str, actor_id: str) -> Agent:
    """Create the Strands host agent with A2A tools for sub-agent delegation."""
    bedrock_model = BedrockModel(model_id=MODEL_ID, region_name=region)
    tools = create_a2a_tools(session_id=session_id, actor_id=actor_id)

    return Agent(
        model=bedrock_model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
    )
