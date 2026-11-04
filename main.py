import argparse

import rerun as rr

from primitives_loader import load_primitives
from poses_loader import load_poses
from scene_builder import build_scene


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D scene animation from CSV primitives and poses."
    )
    parser.add_argument(
        "--primitives",
        default="data/scene/sample_primitives.csv",
        help="CSV with primitive definitions (default: data/scene/sample_primitives.csv)",
    )
    parser.add_argument(
        "--poses",
        default="data/scene/sample_poses.csv",
        help="CSV with timestamped poses (default: data/scene/sample_poses.csv)",
    )
    parser.add_argument(
        "--save",
        metavar="FILE",
        help="Save recording to a .rrd file. Open with 'rerun FILE' or https://app.rerun.io",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Serve the scene in a browser (starts gRPC + web viewer server)",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="HTTP port for --web mode (default: 9090)",
    )
    args = parser.parse_args()

    primitives = load_primitives(args.primitives)
    poses = load_poses(args.poses)
    timestamps = sorted({p.timestamp for p in poses})

    print(f"Primitives: {len(primitives)} ({', '.join(str(k) for k in sorted(primitives))})")
    print(f"Pose records: {len(poses)} over {len(timestamps)} timestamps "
          f"[{timestamps[0]:.3f} .. {timestamps[-1]:.3f}]")

    if args.save:
        rr.init("3d_scene_viewer")
        rr.save(args.save)
        build_scene(primitives, poses)
        print(f"Saved to {args.save}")
        print(f"  Open with:  rerun {args.save}")
        print(f"  Or upload to https://app.rerun.io")

    elif args.web:
        rr.init("3d_scene_viewer")
        grpc_uri = rr.serve_grpc()
        rr.serve_web_viewer(connect_to=grpc_uri, open_browser=False, web_port=args.web_port)
        build_scene(primitives, poses)
        print(f"Open your browser at: http://localhost:{args.web_port}")
        print("Press Ctrl+C to stop.")
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    else:
        rr.init("3d_scene_viewer", spawn=True)
        build_scene(primitives, poses)
        print("Scene logged. Interact with the Rerun viewer.")


if __name__ == "__main__":
    main()
