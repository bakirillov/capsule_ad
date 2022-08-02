"""The script to reproduce Figure 1.
This file is to be ran as a standalone script using 
https://github.com/HarisIqbal88/PlotNeuralNet repository.
"""



import sys
sys.path.append('../')
from pycore.tikzeng import *


PRIM_HEIGHT = 3
PRIM_WIDTH = 3
PRIM_DEPTH = 4

SEC_HEIGHT = 16
SEC_WIDTH = 1
SEC_DEPTH = 4



def my_cor():
    return r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\AnomalyColor{rgb:red,5;black,0.5}
\def\ormalColor{rgb:blue,5;black,0.3}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\AgreementColor{rgb:green,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
"""

# Agreement
def to_Agreement(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+name+""",
        caption="""+ caption +r""",
        fill=\AgreementColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""


# Primary
def to_Primary(
    name, n_filer=16, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" ",
    color="\ConvColor"
):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        fill="""+color+""",
        opacity=0.2,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# Secondary
def to_Secondary(
    name, n_filer=16, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" ",
    color="\AnomalyColor"
):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        ylabel={{"""+ str(n_filer) +"""}},
        fill="""+color+""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""




def to_Empty(
    name, offset="(0,0,0)", to="(0,0,0)", caption=" ",
    color="\AnomalyColor"
):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        fill="""+color+""",
        height=0,
        width=0,
        depth=0
        }
    };
"""


# defined your arch
arch = [
    to_head( '..' ),
    my_cor(),
    to_begin(),
    
    #input
    to_input('picture.png'),
    
    #convolutional layer
    to_Conv("conv", 256, 1, offset="(0,0,0)", to="(0,0,0)", height=28, depth=28, width=2, caption="Preprocessing"),
    to_Agreement(
        "agreement", offset="(7,0,0)", to="(conv-east)", width=1, height=20, depth=20, opacity=0.5, caption="Routing"
    ),
    
    #primary capsules
    to_Conv("primary1", "", "", offset="(5,0.5,0)", to="(0,0,0)", height=PRIM_HEIGHT, depth=PRIM_DEPTH, width=PRIM_WIDTH),
    to_Conv("primary2", "", "", offset="(5,2,0)", to="(0,0,0)", height=PRIM_HEIGHT, depth=PRIM_DEPTH, width=PRIM_WIDTH),
    to_Conv("primary3", "", "", offset="(5,-0.5,0)", to="(0,0,0)", height=PRIM_HEIGHT, depth=PRIM_DEPTH, width=PRIM_WIDTH),
    to_Conv("primary4", "", "", offset="(5,0.5,3)", to="(0,0,0)", height=PRIM_HEIGHT, depth=PRIM_DEPTH, width=PRIM_WIDTH),
    to_Conv("primary5", "", "", offset="(5,2,3)", to="(0,0,0)", height=PRIM_HEIGHT, depth=PRIM_DEPTH, width=PRIM_WIDTH),
    to_Conv(
        "primary6", 32, 256, offset="(5,-0.5,3)", to="(0,0,0)", height=PRIM_HEIGHT, depth=PRIM_DEPTH, width=PRIM_WIDTH,
    ),
    to_Conv("primary7", "", "", offset="(5,1.5,6)", to="(0,0,0)", height=PRIM_HEIGHT, depth=PRIM_DEPTH, width=PRIM_WIDTH),
    to_Conv("primary8", "", "", offset="(5,0,6)", to="(0,0,0)", height=PRIM_HEIGHT, depth=PRIM_DEPTH, width=PRIM_WIDTH),
    to_Primary(
        "all_primaries", 16, offset="(5,1,3)", to="(0,0,0)", height=20, depth=38, width=6,
        caption="Primary capsules"
    ),
    to_connection("conv", "all_primaries"),
    to_connection("all_primaries", "agreement"),
    
    #secondary capsules
    to_Secondary(
        "normal", 16, offset="(11,0.2,3)", to="(0,0,0)", height=SEC_HEIGHT, depth=SEC_DEPTH, width=SEC_WIDTH,
        color="\ormalColor", caption="Normal"
    ),
    to_Secondary(
        "anomaly", 16, offset="(11,0.2,0)", to="(0,0,0)", height=SEC_HEIGHT, depth=SEC_DEPTH, width=SEC_WIDTH,
        color="\AnomalyColor", caption="Anomal"
    ),
    to_Empty("empty", offset="(15,0,6)", to="(0,0,0)"),
    to_connection("agreement", "normal"),
    to_connection("agreement", "anomaly"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
