7.8 instal mmcv
根据错误信息，可以初步判断这个错误可能是由于 CUDA 版本不匹配引起的。建议检查 PyTorch 和 CUDA 的版本是否一致，如果不一致，则需要卸载 PyTorch 和 CUDA，并重新安装相同版本的 PyTorch 和 CUDA。

```
 running build_ext
      C:\conda\lib\site-packages\torch\utils\cpp_extension.py:358: UserWarning: Error checking compiler version for cl: [WinError 2] 系统找不到指定的文件。    
        warnings.warn(f'Error checking compiler version for {compiler}: {error}')
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "C:\Users\lutao\AppData\Local\Temp\pip-install-sx4mcvko\mmcv_cc2595da1c82447691d22c3fa97662ff\setup.py", line 437, in <module>
          setup(
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\__init__.py", line 107, in setup
          return distutils.core.setup(**attrs)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\core.py", line 185, in setup
          return run_commands(dist)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\core.py", line 201, in run_commands
          dist.run_commands()
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\dist.py", line 969, in run_commands
          self.run_command(cmd)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\dist.py", line 1234, in run_command
          super().run_command(command)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\dist.py", line 988, in run_command
          cmd_obj.run()
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\wheel\bdist_wheel.py", line 343, in run
          self.run_command("build")
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\dist.py", line 1234, in run_command
          super().run_command(command)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\dist.py", line 988, in run_command
          cmd_obj.run()
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\command\build.py", line 131, in run
          self.run_command(cmd_name)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\dist.py", line 1234, in run_command
          super().run_command(command)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\dist.py", line 988, in run_command
          cmd_obj.run()
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\command\build_ext.py", line 84, in run
          _build_ext.run(self)
        File "C:\conda\lib\site-packages\Cython\Distutils\old_build_ext.py", line 186, in run
          _build_ext.build_ext.run(self)
        File "C:\Users\lutao\AppData\Roaming\Python\Python310\site-packages\setuptools\_distutils\command\build_ext.py", line 345, in run
          self.build_extensions()
        File "C:\conda\lib\site-packages\torch\utils\cpp_extension.py", line 499, in build_extensions
          _check_cuda_version(compiler_name, compiler_version)
        File "C:\conda\lib\site-packages\torch\utils\cpp_extension.py", line 386, in _check_cuda_version
          raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
      RuntimeError:
      The detected CUDA version (10.1) mismatches the version that was used to compile
      PyTorch (11.7). Please make sure to use the same CUDA versions.
     
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for mmcv
  Running setup.py clean for mmcv
Failed to build mmcv
ERROR: Could not build wheels for mmcv, which is required to install pyproject.toml-based projects

```
RuntimeError: The detected CUDA version (10.1) mismatches the version that was used to compile PyTorch (11.7). Please make sure to use the same CUDA versions.：运行时错误，检测到的 CUDA 版本（10.1）与编译 PyTorch 时使用的版本（11.7）不匹配

` nvcc --version `
10.1
```
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```
11.7
```
import sys

print(sys.executable)
```
C:\conda\python.exe

这个命令会在 CUDA 安装路径下查找以 CUDA 开头的文件夹，并列出它们的名称。同样，每个 CUDA 版本的安装路径都包含版本号信息。

需要注意的是，这些命令只会列出已经安装的 CUDA 运行时版本，不会列出 CUDA 工具包等其他相关组件的版本信息。如果您需要查看完整的 CUDA 版本信息，可以使用 nvcc --version 命令来获取。
```
dir /b /ad "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA*"
```
Gitbash 上的环境报错
```
lutao@DESKTOP-J58VJR4 MINGW64 ~/Desktop/git_AA_NAS/NAS/FNA-master/fna_det (master)
$ sh scripts/arch_adapt_retinanet.sh
./tools/search.py:65: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if args.job_name is '':
./tools/search.py:65: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if args.job_name is '':
./tools/search.py:65: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if args.job_name is '':
./tools/search.py:65: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if args.job_name is '':
./tools/search.py:65: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if args.job_name is '':
./tools/search.py:65: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if args.job_name is '':
./tools/search.py:65: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if args.job_name is '':
./tools/search.py:65: SyntaxWarning: "is" with a literal. Did you mean "=="?
  if args.job_name is '':
Traceback (most recent call last):
  File "./tools/search.py", line 14, in <module>
    import models
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\__init__.py", line 1, in <module>
    from .derived_retinanet_backbone import FNA_Retinanet
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\derived_retinanet_backbone.py", line 3, in <module>
    from mmcv.cnn import kaiming_init
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\__init__.py", line 4, in <module>
    from .image import *
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\__init__.py", line 5, in <module>
    from .geometric import (cutout, imcrop, imflip, imflip_, impad,
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\geometric.py", line 7, in <module>
    from mmengine.utils import to_2tuple
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmengine\__init__.py", line 3, in <module>
    from .config import *
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmengine\config\__init__.py", line 2, in <module>
    from .config import Config, ConfigDict, DictAction, read_base
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmengine\config\config.py", line 20, in <module>
    from yapf.yapflib.yapf_api import FormatCode
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\__init__.py", line 41, in <module>
    from yapf.yapflib import yapf_api
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\yapflib\yapf_api.py", line 39, in <module>
    from yapf.pyparser import pyparser
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\pyparser\pyparser.py", line 44, in <module>
    from yapf.yapflib import format_token
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\yapflib\format_token.py", line 23, in <module>
    from yapf.pytree import pytree_utils
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\pytree\pytree_utils.py", line 30, in <module>
    from yapf_third_party._ylib2to3 import pygram
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf_third_party\_ylib2to3\pygram.py", line 29, in <module>
    python_grammar = driver.load_grammar(_GRAMMAR_FILE)
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf_third_party\_ylib2to3\pgen2\driver.py", line 252, in load_grammar
    g.load(gp)
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf_third_party\_ylib2to3\pgen2\grammar.py", line 95, in load
    d = pickle.load(f)
EOFError: Ran out of input
Traceback (most recent call last):
  File "./tools/search.py", line 14, in <module>
    import models
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\__init__.py", line 1, in <module>
    from .derived_retinanet_backbone import FNA_Retinanet
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\derived_retinanet_backbone.py", line 3, in <module>
    from mmcv.cnn import kaiming_init
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\__init__.py", line 4, in <module>
    from .image import *
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\__init__.py", line 5, in <module>
    from .geometric import (cutout, imcrop, imflip, imflip_, impad,
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\geometric.py", line 7, in <module>
    from mmengine.utils import to_2tuple
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmengine\__init__.py", line 3, in <module>
    from .config import *
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmengine\config\__init__.py", line 2, in <module>
    from .config import Config, ConfigDict, DictAction, read_base
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmengine\config\config.py", line 20, in <module>
    from yapf.yapflib.yapf_api import FormatCode
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\__init__.py", line 41, in <module>
    from yapf.yapflib import yapf_api
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\yapflib\yapf_api.py", line 39, in <module>
    from yapf.pyparser import pyparser
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\pyparser\pyparser.py", line 44, in <module>
    from yapf.yapflib import format_token
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\yapflib\format_token.py", line 23, in <module>
    from yapf.pytree import pytree_utils
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf\pytree\pytree_utils.py", line 30, in <module>
    from yapf_third_party._ylib2to3 import pygram
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf_third_party\_ylib2to3\pygram.py", line 29, in <module>
    python_grammar = driver.load_grammar(_GRAMMAR_FILE)
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf_third_party\_ylib2to3\pgen2\driver.py", line 252, in load_grammar
    g.load(gp)
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\yapf_third_party\_ylib2to3\pgen2\grammar.py", line 95, in load
    d = pickle.load(f)
EOFError: Ran out of input
Traceback (most recent call last):
  File "./tools/search.py", line 14, in <module>
Traceback (most recent call last):
  File "./tools/search.py", line 14, in <module>
    import models
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\__init__.py", line 1, in <module>
    import models
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\__init__.py", line 1, in <module>
    from .derived_retinanet_backbone import FNA_Retinanet
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\derived_retinanet_backbone.py", line 3, in <module>
    from .derived_retinanet_backbone import FNA_Retinanet
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\derived_retinanet_backbone.py", line 3, in <module>
    from mmcv.cnn import kaiming_init
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\__init__.py", line 4, in <module>
    from mmcv.cnn import kaiming_init
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\__init__.py", line 4, in <module>
    from .image import *
    from .image import *  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\__init__.py", line 11, in <module>

  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\__init__.py", line 11, in <module>
    from .photometric import (adjust_brightness, adjust_color, adjust_contrast,
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\photometric.py", line 8, in <module>
    from .photometric import (adjust_brightness, adjust_color, adjust_contrast,
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\photometric.py", line 8, in <module>
    from PIL import Image, ImageEnhance
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\PIL\Image.py", line 100, in <module>
    from PIL import Image, ImageEnhance
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\PIL\Image.py", line 100, in <module>
    from . import _imaging as core
ImportError: DLL load failed while importing _imaging: ▒Ҳ▒▒▒ָ▒▒▒▒ģ▒顣
    from . import _imaging as core
ImportError: DLL load failed while importing _imaging: ▒Ҳ▒▒▒ָ▒▒▒▒ģ▒顣
Traceback (most recent call last):
  File "./tools/search.py", line 14, in <module>
    import models
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\__init__.py", line 1, in <module>
    from .derived_retinanet_backbone import FNA_Retinanet
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\derived_retinanet_backbone.py", line 3, in <module>
    from mmcv.cnn import kaiming_init
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\__init__.py", line 4, in <module>
    from .image import *
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\__init__.py", line 11, in <module>
    from .photometric import (adjust_brightness, adjust_color, adjust_contrast,
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\photometric.py", line 8, in <module>
    from PIL import Image, ImageEnhance
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\PIL\Image.py", line 100, in <module>
    from . import _imaging as core
ImportError: DLL load failed while importing _imaging: ▒Ҳ▒▒▒ָ▒▒▒▒ģ▒顣
Traceback (most recent call last):
  File "./tools/search.py", line 14, in <module>
Traceback (most recent call last):
  File "./tools/search.py", line 14, in <module>
    import models
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\__init__.py", line 1, in <module>
    import models
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\__init__.py", line 1, in <module>
    from .derived_retinanet_backbone import FNA_Retinanet
      File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\derived_retinanet_backbone.py", line 3, in <module>
from .derived_retinanet_backbone import FNA_Retinanet
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\derived_retinanet_backbone.py", line 3, in <module>
        from mmcv.cnn import kaiming_initfrom mmcv.cnn import kaiming_init

  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\__init__.py", line 4, in <module>
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\__init__.py", line 4, in <module>
    from .image import *
from .image import *  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\__init__.py", line 11, in <module>

  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\__init__.py", line 11, in <module>
        from .photometric import (adjust_brightness, adjust_color, adjust_contrast,from .photometric import (adjust_brightness, adjust_color, adjust_contrast,

  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\photometric.py", line 8, in <module>
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\photometric.py", line 8, in <module>
    from PIL import Image, ImageEnhance
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\PIL\Image.py", line 100, in <module>
    from PIL import Image, ImageEnhance
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\PIL\Image.py", line 100, in <module>
    from . import _imaging as core
    from . import _imaging as coreImportError
: ImportError: DLL load failed while importing _imaging: ▒Ҳ▒▒▒ָ▒▒▒▒ģ▒顣
DLL load failed while importing _imaging: ▒Ҳ▒▒▒ָ▒▒▒▒ģ▒顣
Traceback (most recent call last):
  File "./tools/search.py", line 14, in <module>
    import models
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\__init__.py", line 1, in <module>
    from .derived_retinanet_backbone import FNA_Retinanet
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\derived_retinanet_backbone.py", line 3, in <module>
    from mmcv.cnn import kaiming_init
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\__init__.py", line 4, in <module>
    from .image import *
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\__init__.py", line 11, in <module>
    from .photometric import (adjust_brightness, adjust_color, adjust_contrast,
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\mmcv\image\photometric.py", line 8, in <module>
    from PIL import Image, ImageEnhance
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\PIL\Image.py", line 100, in <module>
    from . import _imaging as core
ImportError: DLL load failed while importing _imaging: ▒Ҳ▒▒▒ָ▒▒▒▒ģ▒顣
Traceback (most recent call last):
  File "C:\conda\envs\pytorch4CUDA10\lib\runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\conda\envs\pytorch4CUDA10\lib\runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\torch\distributed\launch.py", line 260, in <module>
    main()
  File "C:\conda\envs\pytorch4CUDA10\lib\site-packages\torch\distributed\launch.py", line 255, in main
    raise subprocess.CalledProcessError(returncode=process.returncode,
subprocess.CalledProcessError: Command '['C:\\conda\\envs\\pytorch4CUDA10\\python.exe', '-u', './tools/search.py', '--local_rank=7', './configs/fna_retinanet_fpn_search.py', '--launcher', 'pytorch', '--seed', '1', '--work_dir', './', '--data_path', './coco/']' returned non-zero exit status 1.
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed.
*****************************************
(pytorch4CUDA10)

```

https://www.yii666.com/blog/211735.html?action=onAll

```
(FNA) C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det>python ./tools/test.py     ./configs/fna_retinanet_fpn_retrain.py     --checkpoint ./retinanet/retin
anet.pth     --net_config ./retinanet/net_config     --data_path ./coco/     --out ./results.pkl     --eval bbox
Traceback (most recent call last):
  File "./tools/test.py", line 9, in <module>
    import models
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\__init__.py", line 1, in <module>
    from .derived_retinanet_backbone import FNA_Retinanet
  File "C:\Users\lutao\Desktop\git_AA_NAS\NAS\FNA-master\fna_det\tools\..\models\derived_retinanet_backbone.py", line 4, in <module>
    from mmdet.models.registry import BACKBONES
  File "C:\conda\envs\FNA\lib\site-packages\mmdet-0.6.0-py3.7.egg\mmdet\__init__.py", line 18, in <module>
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
AssertionError: MMCV==0.2.10 is used but incompatible. Please install mmcv>=2.0.0rc4, <2.1.0.

```