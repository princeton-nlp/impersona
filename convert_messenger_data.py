import json
import datetime
import os
from collections import Counter
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def convert_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp/1000).strftime("%b %d, %Y %I:%M:%S %p")


def process_json_file(input_file, output_dir, me_participant):
    with open(input_file, "r") as f:
        data = json.load(f)

    # Skip conversations with more than 2 participants
    if len(data["participants"]) > 2:
        print(f"Skipping group chat {input_file} - has more than 2 participants")
        return
    
    # Check if the specified participant is in the conversation
    participant_names = [p for p in data["participants"]]
    if me_participant not in participant_names:
        print(f"Skipping {input_file} - {me_participant} is not in the conversation. Only found: {participant_names}")
        return

    thread_name = data["threadName"].replace(" ", "_")
    output_file = output_dir / f"{thread_name}_fb.txt"

    msgs = 0
    with open(output_file, "w") as f:
        for message in data["messages"]:
            msg_text = ""
            timestamp = convert_timestamp(message["timestamp"])
            sender = message["senderName"]
            
            if sender == me_participant:
                sender = "Me"
            
            msg_text += f"{timestamp}\n"
            msg_text += f"{sender}\n"
            
            if message["type"] == "text":
                msg_text += f"{message['text']}\n"
            elif message["type"] == "media":
                continue
            elif message["type"] == "link":
                msg_text += f"{message['text']}\n"
            elif message["type"] == "placeholder":
                msg_text += f"{message['text']}\n"
            else:
                raise ValueError(f"Unknown message type: {message['type']}")
            
            if message["reactions"]:
                msg_text += "Tapbacks:\n"
                for reaction in message["reactions"]:
                    msg_text += f"{reaction['reaction']} by {reaction['actor']}\n"
            msg_text += "\n"

            f.write(msg_text)
            msgs += 1
    if msgs > 0:
        print(f"Wrote {msgs} messages to {output_file}")
    else:
        print(f"Skipping {input_file} - no messages to write")
        os.remove(output_file)


def get_all_participants(input_dir):
    participants = Counter()
    for file in input_dir.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
            participants.update(data["participants"])
    participants, _ = zip(*participants.most_common())
    return list(reversed(participants))


def main(input_dir, output_dir, me_participant):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    all_participants = get_all_participants(input_dir)
    if me_participant not in all_participants:
        raise ValueError(f"Your name {me_participant} is not in any conversations. Please select a different name from the following participants: {', '.join(all_participants)}")

    for file in input_dir.glob("*.json"):
        process_json_file(file, output_dir, me_participant)

if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Facebook Messenger data to iMessage data", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input_dir", type=str, default="messenger_export", help="Path to the directory containing the Facebook Messenger data"
        )
    parser.add_argument(
        "--output_dir", type=str, default="data/imessage_export", help="Path to the directory to save the converted iMessage data"
        )
    parser.add_argument(
        "--me_participant", type=str, required=True, help="Your name as it appears in the Facebook Messenger data"
        )
    args = parser.parse_args()
    main(**vars(args))
