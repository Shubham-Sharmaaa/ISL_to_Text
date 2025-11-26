# import ezdxf

# # Create new DXF document
# doc = ezdxf.new(dxfversion="R2010")
# msp = doc.modelspace()

# # Helper function to draw rectangle with label
# def draw_room(x, y, w, h, label):
#     # Rectangle
#     msp.add_lwpolyline([(x,y), (x+w,y), (x+w,y+h), (x,y+h)], close=True)
#     # Label text at center
#     msp.add_text(label, dxfattribs={'height': 12}).set_pos((x+w/2, y+h/2), align='MIDDLE_CENTER')

# # ================= FLOOR PLAN ==================
# # Units: 1 unit = 12 inches (1 foot) approx, adjust as needed

# # Shops & Porch (Front side)
# draw_room(0,0,216,132, "SHOP 18'x11'")    # Left shop
# draw_room(216,0,156,204, "PORCH 13'x17'") # Porch
# draw_room(372,0,108,132, "SHOP 9'x11'")   # Right shop

# # Lawn
# draw_room(372,132,108,72, "FRONT LAWN 9'6\" wide")

# # Kitchen + Utility
# draw_room(0,132,144,150, "KITCHEN 10'x12'6\" + Utility")

# # Lobby + Dining
# draw_room(144,204,228,180, "LOBBY 12'x14' + DINING")

# # Drawing Room
# draw_room(372,204,168,168, "DRAWING ROOM 12'x14'")

# # Bedrooms
# draw_room(0,282,168,168, "BED ROOM 11'x13'")
# draw_room(168,384,168,168, "MASTER BED 14'x14' + Dress")
# draw_room(336,384,168,168, "BED ROOM 11'x14'")

# # Toilets + Duct (Top Row)
# draw_room(0,450,120,120, "TOILET 8'x10'")
# draw_room(120,450,120,120, "DUCT 7'6\"x6'")
# draw_room(240,450,120,120, "TOILET 8'x10'")

# # ===============================================

# # Save DXF file
# doc.saveas("redesigned_floor_plan.dxf")
# print("DXF Floor Plan saved as redesigned_floor_plan.dxf")
import ezdxf
from ezdxf.lldxf.const import Units

doc = ezdxf.new(dxfversion="R2010")
doc.units = Units.INCH  # you’re drawing in inches
msp = doc.modelspace()

# -------- Layers --------
def add_layer(name, color=7, ltype="CONTINUOUS"):
    if name not in doc.layers:
        doc.layers.add(name, color=color, linetype=ltype)

add_layer("WALLS", color=7)
add_layer("ROOMS", color=3)
add_layer("TEXT",  color=2)
add_layer("OPENINGS", color=1)

# -------- Utilities --------
def rect_pts(x, y, w, h):
    return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]

def draw_rect(x, y, w, h, layer="ROOMS", closed=True):
    return msp.add_lwpolyline(rect_pts(x, y, w, h), close=closed, dxfattribs={"layer": layer})

def offset_rect(x, y, w, h, t):
    """Returns outer and inner rectangles to visualize wall thickness t (inches)."""
    outer = rect_pts(x, y, w, h)
    inner = rect_pts(x+t, y+t, w-2*t, h-2*t)
    return outer, inner

def draw_room(x, y, w, h, label, wall_thk=6):
    # walls: double polyline (outer+inner)
    out_pts, in_pts = offset_rect(x, y, w, h, wall_thk)
    msp.add_lwpolyline(out_pts, close=True, dxfattribs={"layer":"WALLS"})
    msp.add_lwpolyline(in_pts,  close=True, dxfattribs={"layer":"WALLS"})
    # room fill line (optional lightweight outline for selection)
    msp.add_lwpolyline(rect_pts(x+wall_thk, y+wall_thk, w-2*wall_thk, h-2*wall_thk),
                       close=True, dxfattribs={"layer":"ROOMS"})
    # centered MText label
    cx, cy = x + w/2, y + h/2
    msp.add_mtext(label, dxfattribs={"layer":"TEXT", "width": w-2*wall_thk, "char_height":12}) \
       .set_location((cx, cy), attachment_point=5)  # 5=MIDDLE_CENTER

# ================= FLOOR PLAN ==================
# Units: 1 unit = 1 inch (12 inches = 1 ft)

# Shops & Porch (Front side)
draw_room(0, 0, 216, 132, "SHOP 18' x 11'")
draw_room(216, 0, 156, 204, "PORCH 13' x 17'")
draw_room(372, 0, 108, 132, "SHOP 9' x 11'")

# Lawn (width corrected to 9'6" = 114 in)
draw_room(372, 132, 114, 72, "FRONT LAWN 9'6\" wide")

# Kitchen + Utility
draw_room(0, 132, 144, 150, "KITCHEN 10' x 12'6\" + Utility")

# Lobby + Dining
draw_room(144, 204, 228, 180, "LOBBY 12' x 14' + DINING")

# Drawing Room
draw_room(372, 204, 168, 168, "DRAWING ROOM 12' x 14'")

# Bedrooms
draw_room(0, 282, 168, 168, "BED ROOM 11' x 13'")
draw_room(168, 384, 168, 168, "MASTER BED 14' x 14' + Dress")
draw_room(336, 384, 168, 168, "BED ROOM 11' x 14'")

# Toilets + Duct (Top Row)
draw_room(0,   450, 120, 120, "TOILET 8' x 10'")
draw_room(120, 450, 120, 120, "DUCT 7'6\" x 6'")
draw_room(240, 450, 120, 120, "TOILET 8' x 10'")

# Save DXF
doc.saveas("redesigned_floor_plan_layered.dxf")
print("Saved: redesigned_floor_plan_layered.dxf")
