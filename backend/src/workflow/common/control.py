import json
import argparse
from src.db.checkpoint import memory
from src.logger import setup_logger
from src.workflow.workflows.estimation.graph import workflow

logger = setup_logger(__name__)

app = workflow.compile(checkpointer=memory)

def parse_args():
    parser = argparse.ArgumentParser(description="Workflow Control CLI")

    parser.add_argument("--thread_id", type=str, required=True, help="Thread ID to resume")
    parser.add_argument("--state_step", type=str, required=True, help="Node to resume from")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["rerun", "check"],
        default="rerun",
        help="Action to perform"
    )

    return parser.parse_args()


def check_prev_result(thread_id: str, state_step: str):
    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Fetching previous state for thread_id={thread_id}")

    snapshot = app.get_state(config)

    if not snapshot.values:
        logger.error("No data found for this thread ID.")
        return

    logger.info("Previous run results retrieved")

    data = snapshot.values.get(state_step, {})
    data_str = json.dumps(data, indent=2)

    logger.debug(f"Preview (first 1000 chars):\n{data_str[:1000]}")

    with open(f"{thread_id}_{state_step}_bom_output.txt", "w", encoding="utf-8") as f:
        f.write(data_str)

    logger.info("Full output written to bom_output.txt")


def rerun_from_the_node(thread_id: str, state_step: str):
    logger.info(f"Rerunning workflow from node: {state_step}")

    config = {"configurable": {"thread_id": thread_id}}

    app.update_state(
        config,
        {},
        as_node=state_step
    )

    logger.info("Execution resumed")

    for event in app.stream(None, config=config):
        for node_name, state_update in event.items():
            logger.info(f"Running Node: {node_name}")

            if node_name == "agent_4_merger":
                logger.info("Agent merger output:")
                logger.info(json.dumps(state_update, indent=2))


if __name__ == "__main__":
        args = parse_args()

        thread_id = args.thread_id
        state_step = args.state_step
        mode = args.mode

        if mode == "check":
            check_prev_result(thread_id, state_step)
        else:
            rerun_from_the_node(thread_id, state_step)
