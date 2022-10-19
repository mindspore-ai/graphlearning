"""version info and check."""
# pylint: disable=C0111
__version__ = 'master'


def mindspore_version_check():
    """
    Do the MindSpore version check for MindSpore Graph Learning. If the
    MindSpore can not be imported, it will raise ImportError. If its
    version is not compatibale with current MindSpore Graph Learning verision,
    it will print a warning.

    Raise:
        ImportError: If the MindSpore can not be imported.
    """

    try:
        import mindspore as ms
        from mindspore import log as logger
    except (ImportError, ModuleNotFoundError):
        print("Can not find MindSpore in current environment. Please install "
              "MindSpore before using MindSpore Graph Learning, by following "
              "the instruction at https://www.mindspore.cn/install")
        raise

    ms_gl_version_match = {'0.1': '1.6.1',
                           'master': '1.7.1'}

    ms_version = ms.__version__[:5]
    required_mindspore_verision = ms_gl_version_match[__version__]

    if ms_version < required_mindspore_verision:
        logger.warning("Current version of MindSpore is not compatible with MindSpore Graph Learning. "
                       "Some functions might not work or even raise error. Please install MindSpore "
                       "version == {} For more details about dependency setting, please check "
                       "the instructions at MindSpore official website https://www.mindspore.cn/install "
                       "or check the README.md at https://gitee.com/mindspore/graphlearning"
                       .format(required_mindspore_verision))
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logger.warning(
                f"Please pay attention to the above warning, countdown: {i}")
            time.sleep(1)
