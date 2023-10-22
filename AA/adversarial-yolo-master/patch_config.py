from torch import optim


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "inria/Train/pos"
        self.lab_dir = "inria/Train/pos/yolo-labels"
        self.cfgfile = "cfg/yolo.cfg"
        self.weightfile = "weights/yolo.weights"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 300

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 20

        self.none_squ_list = ['A4RealWorld', 'ipad_tst']        # 非正方形的配置文件

        self.loss_target = lambda obj, cls: obj * cls


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'

class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"

class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls




class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 56    # 3090 24G 56
        self.patch_size = 6

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj


class TestCLS(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 128    # 3090 24G 56
        self.patch_size = 6

        self.patch_name = 'TestCLS'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj


class A4RealWorldObj(BaseConfig):
    """
    Using this patch to test in real world in A4 printed papers.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 32    # 3090 24G 56
        self.patch_size = 300
        self.patch_size_x = 210     # A4 paper without thinking print area
        self.patch_size_y = 297

        self.patch_name = 'A4RealWorld'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

class iPadObj(BaseConfig):
    """
    Using this patch to test in real world in A4 printed papers.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 56    # 3090 24G 56
        self.patch_size = 300
        self.patch_size_x = 160     # A4 paper without thinking print area
        self.patch_size_y = 200

        self.patch_name = 'ipad_tst'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj


patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj,
    "A4RealWorld": A4RealWorldObj,
    "testing_exp_cls" : TestCLS,
    "ipad_tst" : iPadObj
}
