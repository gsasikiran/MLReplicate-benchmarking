import os
import subprocess

def run_and_print(args):
    result = subprocess.run(args, capture_output=True, text=True)
    print(f"[命令] {' '.join(args)}")
    print(result.stdout)
    print(result.stderr)
    return result

def compile_latex_project(project_dir, main_tex_file):
    """编译 LaTeX 项目 using Tectonic"""
    import os
    import subprocess
    
    # Save original directory BEFORE changing
    original_dir = os.getcwd()
    
    try:
        os.chdir(project_dir)
        base_name = main_tex_file.replace('.tex', '')
        
        print(f"开始编译: {main_tex_file}")
        # Tectonic handles everything in one command (no separate bibtex step needed)
        result = subprocess.run(['tectonic', main_tex_file], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"成功: PDF 已生成")
            return True
        else:
            print(f"编译失败:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except FileNotFoundError:
        print(f"发生错误: Tectonic not found. Install with: conda install -c conda-forge tectonic")
        return False
    except Exception as e:
        print(f"发生错误: {e}")
        return False
        
    finally:
        # Return to original directory
        os.chdir(original_dir)