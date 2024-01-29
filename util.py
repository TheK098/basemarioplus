import warnings


def ignoreWarnings():
    # Ignore specific deprecation warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, 
                            message=".*old step API which returns one bool instead of two.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, 
                            message=".*The argument mode in render method is deprecated.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, 
                            message=".*The torch.cuda.*DtypeTensor constructors are no longer recommended.*")

    # Ignore user warnings
    warnings.filterwarnings("ignore", category=UserWarning, 
                            message=".*The environment SuperMarioBros-1-1-v0 is out of date.*")
    warnings.filterwarnings("ignore", category=UserWarning, 
                            message=".*No render modes was declared in the environment.*")