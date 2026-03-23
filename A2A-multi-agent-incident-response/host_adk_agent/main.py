import logging
from dotenv import load_dotenv
from bedrock_agentcore import BedrockAgentCoreApp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = BedrockAgentCoreApp()

host_agent = None


@app.entrypoint
async def call_agent(payload: dict, context):
    global host_agent

    session_id = context.session_id
    logger.info(f"Received request with session_id: {session_id}")

    actor_id = context.request_headers[
        "x-amzn-bedrock-agentcore-runtime-custom-actorid"
    ]

    if not actor_id:
        raise Exception("Actor id is not set")

    if not session_id:
        raise Exception("Context session_id is not set")

    if not host_agent:
        # Import agent creation inside entrypoint so workload identity is available
        from agent import create_host_agent, fetch_agents_cards

        logger.info("Initializing host agent and fetching agent cards...")
        try:
            host_agent = create_host_agent(
                session_id=session_id, actor_id=actor_id
            )
            agents_cards = fetch_agents_cards()
            logger.info(
                f"Successfully initialized host agent. Agent cards: {list(agents_cards.keys())}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize host agent: {e}", exc_info=True)
            raise

        yield agents_cards

    query = payload.get("prompt")
    logger.info(f"Processing query: {query}")

    if not query:
        raise KeyError("'prompt' field is required in payload")

    # Stream response from Strands agent
    async for event in host_agent.stream_async(query):
        yield event


if __name__ == "__main__":
    app.run()  # Ready to run on Bedrock AgentCore
