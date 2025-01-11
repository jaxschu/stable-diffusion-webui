from modules import launch_utils
from transformers import CLIPProcessor, CLIPModel
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""


# ========== CLIP模型加载 ========== #
clip_model = None
clip_processor = None

def load_clip():
    global clip_model, clip_processor
    model_path = os.path.join("models", "clip-vit-large-patch14")
    
    try:
        # 加载CLIP模型和处理器
        clip_model = CLIPModel.from_pretrained(model_path, local_files_only=True)
        clip_processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        print("CLIP 模型加载成功！")
    except Exception as e:
        print(f"CLIP 模型加载失败: {e}")

# ========== 原始的 WebUI 启动配置 ========== #
args = launch_utils.args
python = launch_utils.python
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos

commit_hash = launch_utils.commit_hash
git_tag = launch_utils.git_tag

run = launch_utils.run
is_installed = launch_utils.is_installed
repo_dir = launch_utils.repo_dir

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
git_clone = launch_utils.git_clone
git_pull_recursive = launch_utils.git_pull_recursive
list_extensions = launch_utils.list_extensions
run_extension_installer = launch_utils.run_extension_installer
prepare_environment = launch_utils.prepare_environment
configure_for_tests = launch_utils.configure_for_tests
start = launch_utils.start

# ========== 主启动程序 ========== #
def main():
    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()
        print(f"Sysinfo saved as {filename}. Exiting...")
        exit(0)

    launch_utils.startup_timer.record("initial startup")

    with launch_utils.startup_timer.subcategory("prepare environment"):
        if not args.skip_prepare_environment:
            prepare_environment()

    # 加载 CLIP 模型
    load_clip()

    if args.test_server:
        configure_for_tests()

    start()

if __name__ == "__main__":
    main()
