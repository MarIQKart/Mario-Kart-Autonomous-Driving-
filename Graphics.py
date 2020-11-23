from PIL import ImageDraw
from PIL import ImageOps

def clamp(val, lower, upper):
    if val < lower:
        val = lower
    if val > upper:
        val = upper
    return val

def clampRGB(r, g, b):
    r = clamp(r, 0, 255)
    g = clamp(g, 0, 255)
    b = clamp(b, 0, 255)
    return r, g, b

def rgb2Hex(r, g, b):
    return "#{0:02x}{1:02x}{2:02x}".format(r, g, b)

def addRect(img, x, y, w, h, r, g, b, border = 1, swap = True):
    r, g, b = clampRGB(r, g, b)
    if swap:
        r, b = b, r

    for i in range(x, x + w):
        for k in range(border):
            img.putpixel((i, y + k), (r, g, b))
            img.putpixel((i, y + h + k), (r, g, b))
    for j in range(y, y + h + border):
        for k in range(border):
            img.putpixel((x + k, j), (r, g, b))
            img.putpixel((x + w + k, j), (r, g, b))

def fillRect(img, x, y, w, h, r, g, b, swap = True):
    r, g, b = clampRGB(r, g, b)
    if swap:
        r, b = b, r

    for i in range(x, x + w):
        for j in range(y, y + h):
            img.putpixel((i, j), (r, g, b))

def drawLine(img, x1, y1, x2, y2, r, g, b, thickness = 1, swap = True):
    r, g, b = clampRGB(r, g, b)
    if swap:
        r, b = b, r

    draw = ImageDraw.Draw(img)
    fillColor = rgb2Hex(r, g, b)
    draw.line((x1, y1, x2, y2), fill = fillColor, width = thickness)

def toGrayscale(img):
    gs = ImageOps.grayscale(img)
    return gs