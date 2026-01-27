import argparse
import sys
import os
import subprocess

def run_module(module_name, args=None):
    """Run a project module as a script."""
    cmd = [sys.executable, "-m", module_name]
    if args:
        cmd.extend(args)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[Error] Command failed: {' '.join(cmd)}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\n[Stopped] Operation cancelled by user.")

def cmd_manage(args):
    """Manage actions and dataset statistics."""
    pass_args = []
    if args.list: pass_args.append("--list")
    if args.add: pass_args.extend(["--add", args.add])
    if args.remove: pass_args.extend(["--remove", args.remove])
    run_module("app.training.manage_dataset", pass_args)

def cmd_collect(args):
    """Direct landmark collection."""
    pass_args = ["--modality", args.modality, "--action", args.action, "--samples", str(args.samples)]
    run_module("app.training.collect_data", pass_args)

def cmd_record(args):
    """Raw data recording."""
    if args.modality == "camera":
        pass_args = ["--action", args.action, "--samples", str(args.samples)]
        run_module("app.training.record_videos", pass_args)
    else:
        pass_args = ["--action", args.action, "--samples", str(args.samples)]
        run_module("app.training.record_glove", pass_args)

def cmd_ingest(args):
    """Ingest raw data."""
    if args.modality == "camera":
        run_module("app.training.ingest_videos")
    else:
        run_module("app.training.ingest_glove")

def cmd_augment(args):
    """Augment dataset."""
    run_module("app.training.augment_data", ["--modality", args.modality])

def cmd_train(args):
    """Train models."""
    if args.modality == "camera":
        run_module("app.training.train_camera")
    else:
        run_module("app.training.train_glove")

def cmd_export(args):
    """Export to TF-Lite."""
    run_module("scripts.export_tflite")

def cmd_migrate(args):
    """Migrate and consolidate raw data."""
    pass_args = []
    if args.undo: pass_args.append("--undo")
    if args.mark_done: pass_args.append("--mark-done")
    run_module("scripts.migrate_data", pass_args)

def cmd_profile(args):
    """Profile inference performance."""
    run_module("app.tools.profiler", ["--mode", "benchmark", "--frames", str(args.frames)])

def cmd_batch_run(args):
    """Run batch (offline) inference."""
    run_module("app.tools.profiler", ["--mode", "batch"])

def cmd_run(args):
    """Run real-time inference."""
    pass_args = ["--mode", args.modality]
    if args.headless: pass_args.append("--headless")
    run_module("app.main", pass_args)

def cmd_install_service(args):
    """Install systemd service."""
    pass_args = []
    if args.install: pass_args.append("--install")
    run_module("app.services.linux_service", pass_args)

def main():
    parser = argparse.ArgumentParser(
        description="SIBI Platform: Unified Control Interface",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Manage
    p_manage = subparsers.add_parser("dataset", help="Manage actions and view statistics")
    p_manage.add_argument("--list", action="store_true", help="List actions and sample counts")
    p_manage.add_argument("--add", type=str, help="Add a new action")
    p_manage.add_argument("--remove", type=str, help="Remove an action")

    # Collect
    p_collect = subparsers.add_parser("collect", help="Direct landmark collection (Camera/Glove)")
    p_collect.add_argument("--modality", choices=["camera", "glove", "both"], default="camera")
    p_collect.add_argument("--action", required=True, help="Action name to record")
    p_collect.add_argument("--samples", type=int, default=10, help="Number of samples")

    # Record
    p_record = subparsers.add_parser("record", help="Raw data recording (AVI/CSV)")
    p_record.add_argument("--modality", choices=["camera", "glove"], default="camera")
    p_record.add_argument("--action", required=True, help="Action name to record")
    p_record.add_argument("--samples", type=int, default=5, help="Number of samples")

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Process raw data into landmarks/npy")
    p_ingest.add_argument("--modality", choices=["camera", "glove"], default="camera")

    # Augment
    p_augment = subparsers.add_parser("augment", help="Data augmentation")
    p_augment.add_argument("--modality", choices=["camera", "glove"], default="camera")

    # Train
    p_train = subparsers.add_parser("train", help="Train modality-specific models")
    p_train.add_argument("--modality", choices=["camera", "glove"], default="camera")

    # Export
    subparsers.add_parser("export", help="Export models to TF-Lite")

    # Migrate
    p_migrate = subparsers.add_parser("migrate", help="Consolidate and rename raw data")
    p_migrate.add_argument("--undo", action="store_true", help="Remove _done suffix")
    p_migrate.add_argument("--mark-done", action="store_true", help="Mark all files as ingested manually")

    # Profile
    p_profile = subparsers.add_parser("profile", help="Profile inference performance")
    p_profile.add_argument("--frames", type=int, default=100, help="Number of frames to profile")
    p_profile.add_argument("--mode", choices=["benchmark", "batch"], default="benchmark", help="Profiling mode")

    # Batch Run
    subparsers.add_parser("batch-run", help="Run batch (offline) inference")

    # Run
    p_run = subparsers.add_parser("run", help="Run real-time inference")
    p_run.add_argument("--modality", choices=["camera", "glove"], default="camera")
    p_run.add_argument("--headless", action="store_true", help="Run in headless mode (no UI)")

    # Install Service
    p_service = subparsers.add_parser("install-service", help="Generate/Install systemd service (Linux)")
    p_service.add_argument("--install", action="store_true", help="Install to user systemd directory")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Map commands to functions
    cmd_map = {
        "dataset": cmd_manage,
        "collect": cmd_collect,
        "record": cmd_record,
        "ingest": cmd_ingest,
        "augment": cmd_augment,
        "train": cmd_train,
        "export": cmd_export,
        "migrate": cmd_migrate,
        "profile": cmd_profile,
        "batch-run": cmd_batch_run,
        "batch-run": cmd_batch_run,
        "run": cmd_run,
        "install-service": cmd_install_service
    }

    if args.command in cmd_map:
        cmd_map[args.command](args)

if __name__ == "__main__":
    main()
