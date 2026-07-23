# Headless Read(image) + Read(trimap) -> MEMatte -> Write render.
#
#   HYP_INPUT=plate.png HYP_TRIMAP=trimap.png HYP_OUTPUT=out.exr \
#     MEMATTE_MODEL_DIR=/path/to/models \
#     HYP_ACESCG=0 HYP_TOKENS=18500 HYP_OUTMODE=0 \
#     OFX_PLUGIN_PATH="$HOME/OFX/Plugins" \
#     NatronRenderer --clear-openfx-cache -t tests/natron/render_matte.py < /dev/null
#
# HYP_OUTMODE: 0 = matte, 1 = RGBA unpremultiplied, 2 = RGBA premultiplied.
import os
import sys

PLUGIN_ID = "com.tokgan.openfx.MEMatte"

inp = os.environ["HYP_INPUT"]
out = os.environ["HYP_OUTPUT"]
trimap = os.environ.get("HYP_TRIMAP", "")

try:
    from NatronEngine import natron  # noqa: F401
except Exception as e:
    print("IMPORT_ERROR:", repr(e))

app = app1  # noqa: F821


def setp(node, name, value):
    p = node.getParam(name)
    if p is None:
        print("NO_PARAM:", name)
        return
    p.setValue(value)


def identity_ocio(node):
    """Make the node's OCIO output space equal its input space, so the matte is
    read/written as raw data rather than being tone-mapped or clamped."""
    try:
        pin = node.getParam("ocioInputSpace")
        pout = node.getParam("ocioOutputSpace")
        if pin and pout:
            v = pin.getValue()
            pout.setValue(v)
            print("OCIO identity on", node.getScriptName(), "space=", v)
    except Exception as e:
        print("OCIO set failed:", repr(e))


reader = app.createReader(inp)
mematte = app.createNode(PLUGIN_ID)
writer = app.createWriter(out)
if reader is None or mematte is None or writer is None:
    print("RESULT: FAIL (node creation)")
    sys.exit(1)

identity_ocio(reader)
identity_ocio(writer)

setp(mematte, "inputIsACEScg", os.environ.get("HYP_ACESCG", "0") == "1")
setp(mematte, "maxTokens", int(os.environ.get("HYP_TOKENS", "18500")))
setp(mematte, "outputMode", int(os.environ.get("HYP_OUTMODE", "0")))
setp(mematte, "modelVariant", int(os.environ.get("HYP_VARIANT", "0")))
setp(mematte, "device", int(os.environ.get("HYP_DEVICE", "0")))
setp(mematte, "trimapEncoding", int(os.environ.get("HYP_TRIENC", "0")))
if os.environ.get("HYP_BUDGET_MB"):
    setp(mematte, "autoBudget", False)
    setp(mematte, "memBudgetMB", int(os.environ["HYP_BUDGET_MB"]))

mematte.connectInput(0, reader)
if trimap:
    tri_reader = app.createReader(trimap)
    if tri_reader is None:
        print("RESULT: FAIL (trimap reader)")
        sys.exit(1)
    # The trimap must NOT be colour-managed: it is label data, not an image.
    identity_ocio(tri_reader)
    mematte.connectInput(1, tri_reader)
else:
    # No trimap clip: fall back to the source alpha.
    setp(mematte, "trimapSource", 1)

writer.connectInput(0, mematte)

first = int(os.environ.get("HYP_FIRST", "1"))
last = int(os.environ.get("HYP_LAST", "1"))
try:
    app.render(writer, first, last)
except Exception:
    app.render([(writer, first, last)])

exists = os.path.exists(out) or last > first
print("RENDERED_RANGE:", first, last)
print("RESULT:", "PASS" if exists else "FAIL")
try:
    natron.quitApplication()
except Exception:
    sys.exit(0)
