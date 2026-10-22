from catcam.app import main
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="本地猫咪饮水监控")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--preview-only", action="store_true", help="仅预览真实视频，不识别、不录制、不发邮件")
    arguments = parser.parse_args()
    main(arguments.config, preview_only=arguments.preview_only)
