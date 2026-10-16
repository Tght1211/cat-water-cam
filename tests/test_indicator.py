import numpy as np
from catcam.indicator import IndicatorMonitor


def test_user_roi_steady_blink_and_off(tmp_path):
    m=IndicatorMonitor(tmp_path)
    m.set_roi([.25,.25,.75,.75])
    on=np.full((60,80,3),80,dtype=np.uint8)
    on[22:36,30:46]=[0,0,240]
    off=np.full_like(on,80)
    for i in range(41):m.observe(100+i*.25,on,True,enabled=True)
    assert m.status(110)['state']=='steady_red'
    for i in range(49):m.observe(111+i*.25,on if (i//2)%2 else off,True,enabled=True)
    assert m.status(123)['state']=='blinking_red'
    for i in range(49):m.observe(124+i*.25,off,True,enabled=True)
    assert m.status(136)['state']=='off'
    m.observe(137,on,False,enabled=True)
    assert m.status(137)['state']=='unknown'
    assert IndicatorMonitor(tmp_path).roi==[.25,.25,.75,.75]


def test_disabled_and_covered_lights_do_not_infer_missing_water(tmp_path):
    m=IndicatorMonitor(tmp_path)
    m.set_roi([0,0,1,1])
    red=np.full((50,50,3),[0,0,255],dtype=np.uint8)
    m.observe(100,red,True,enabled=False)
    assert m.status(100)['state']=='unconfigured'
    m.observe(101,red,True,True,True)
    assert m.status(101)['state']=='unknown'
