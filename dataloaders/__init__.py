from .midair import DataLoaderMidAir as MidAir
from .kitti import DataLoaderKittiRaw as KittiRaw
from .tartanair import DataLoaderTartanAir as TartanAir
from .generic import DataloaderParameters

def get_loader(name : str):
    available = {
        "midair"        : MidAir(),
        "kitti-raw"     : KittiRaw(),
        "tartanair"   : TartanAir()
    }
    try:
        return available[name]
    except:
        print("Dataloaders available:")
        print(available.keys())
        raise NotImplementedError
