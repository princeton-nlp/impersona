from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import shutil
from pathlib import Path


def load_convo_files(convo_dir: Path):
    convo_files = list(convo_dir.glob("*.txt"))
    print(f"Found {len(convo_files)} conversation files in {convo_dir.as_posix()}")
    return convo_files


def main(
    mac_export_dir: str,
    ios_export_dir: str,
    output_dir: str,
):
    mac_export_dir = Path(mac_export_dir)
    ios_export_dir = Path(ios_export_dir)
    output_dir = Path(output_dir)

    if output_dir.exists() and len(list(output_dir.glob("*.txt"))) > 0:
        raise FileExistsError(f"Output file {output_dir.as_posix()} already exists. Please delete it before running this script.")

    if not mac_export_dir.exists() or not mac_export_dir.is_dir():
        raise FileNotFoundError(f"Mac export directory {mac_export_dir.as_posix()} does not exist. Please run the mac export script first.")
    if not ios_export_dir.exists() or not ios_export_dir.is_dir():
        raise FileNotFoundError(f"iOS export directory {ios_export_dir.as_posix()} does not exist. Please run the ios export script first.")

    mac_convo_files = load_convo_files(mac_export_dir)
    ios_convo_files = load_convo_files(ios_export_dir)
    
    mac_convo_filenames = {x.name for x in mac_convo_files}
    ios_convo_filenames = {x.name for x in ios_convo_files}
    common_convo_filenames = mac_convo_filenames & ios_convo_filenames
    mac_only_convo_filenames = mac_convo_filenames - ios_convo_filenames
    ios_only_convo_filenames = ios_convo_filenames - mac_convo_filenames
    all_convo_filenames = mac_convo_filenames | ios_convo_filenames

    print(f"Found {len(all_convo_filenames)} conversation files in {mac_export_dir.as_posix()} and {ios_export_dir.as_posix()}")
    print(f"Found {len(common_convo_filenames)} common conversation files (taking longest)")
    print(f"Found {len(mac_only_convo_filenames)} conversation files only in mac")
    print(f"Found {len(ios_only_convo_filenames)} conversation files only in ios")

    files_to_take_from_mac = mac_only_convo_filenames
    files_to_take_from_ios = ios_only_convo_filenames
    
    for convo_filename in common_convo_filenames:
        mac_convo_file = mac_export_dir / convo_filename
        ios_convo_file = ios_export_dir / convo_filename
        mac_convo_file_size = mac_convo_file.stat().st_size
        ios_convo_file_size = ios_convo_file.stat().st_size
        if mac_convo_file_size > ios_convo_file_size:
            files_to_take_from_mac.add(convo_filename)
        else:
            files_to_take_from_ios.add(convo_filename)
    
    print(f"Taking {len(files_to_take_from_mac)} unique conversation files from mac")
    print(f"Taking {len(files_to_take_from_ios)} unique conversation files from ios")
    print(f"Total files to take: {len(files_to_take_from_mac) + len(files_to_take_from_ios)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for convo_filename in files_to_take_from_mac:
        shutil.copy(mac_export_dir / convo_filename, output_dir / convo_filename)
    for convo_filename in files_to_take_from_ios:
        shutil.copy(ios_export_dir / convo_filename, output_dir / convo_filename)

    print(f"Merged {len(files_to_take_from_mac) + len(files_to_take_from_ios)} conversation files into {output_dir.as_posix()}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Merge imessage exports", formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mac_export_dir", type=str, help="Path to the mac export", default="imessage_export_mac")
    parser.add_argument("--ios_export_dir", type=str, help="Path to the ios export", default="imessage_export_ios")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory", default="imessage_export_mac_and_ios")
    args = parser.parse_args()
    main(**vars(args))
