import gmsh
import sys


def patt_gate(Lx, Ly, w, r, d, h, diam, lc, c):
    g = gmsh.model.geo
    lx2 = (Lx-d-2*h) / 2    
    ly2 = (Ly-w) / 2
    w2 = (w-r) / 2

    g.addPoint(c[0], c[1] + ly2, c[2], lc, 110)
    g.addPoint(c[0] + lx2, c[1] + ly2, c[2], lc , 120)  # lc/2
    g.addPoint(c[0] + lx2 + h, c[1] + ly2 + w2, c[2], lc , 130)  # lc/2

    g.addPoint(c[0]+Lx, c[1] + ly2, c[2], lc, 210)
    g.addPoint(c[0]+Lx - lx2, c[1] + ly2, c[2], lc , 220)  # lc/2
    g.addPoint(c[0]+Lx - lx2 - h, c[1] + ly2 + w2, c[2], lc , 230)  # lc/2

    g.addPoint(c[0]+Lx, c[1]+Ly - ly2, c[2], lc, 310)
    g.addPoint(c[0]+Lx - lx2, c[1]+Ly - ly2, c[2], lc / 2, 320)  # lc/2
    g.addPoint(c[0]+Lx - lx2 - h, c[1]+Ly - ly2 - w2, c[2], lc , 330)  # lc/2

    g.addPoint(c[0], c[1]+Ly - ly2, c[2], lc, 410)
    g.addPoint(c[0] + lx2, c[1]+Ly - ly2, c[2], lc , 420)  # lc/2
    g.addPoint(c[0] + lx2 + h, c[1]+Ly - ly2 - w2, c[2], lc , 430)  # lc/2

    # Creating the center hole
    g.addPoint(c[0]+Lx/2, c[1]+Ly/2, c[2], lc/2, 50)
    g.addPoint(c[0]+Lx/2, c[1]+Ly/2+diam/2, c[2], lc/2, 510)
    g.addPoint(c[0]+Lx/2+diam/2, c[1]+Ly/2, c[2], lc/2, 511)
    g.addPoint(c[0]+Lx/2, c[1]+Ly/2-diam/2, c[2], lc/2, 512)
    g.addPoint(c[0]+Lx/2-diam/2, c[1]+Ly/2, c[2], lc/2, 520)


    g.addLine(15, 16, 101)
    g.addLine(16, 210)
    g.addLine(210, 220)
    g.addLine(220, 230)
    g.addLine(230, 330)
    g.addLine(330, 320)
    g.addLine(320, 310)
    g.addLine(310, 17) 
    g.addLine(17, 18)  # 109
    g.addLine(18, 410)
    g.addLine(410, 420)
    g.addLine(420, 430)
    g.addLine(430, 130)
    g.addLine(130, 120)
    g.addLine(120, 110)
    g.addLine(110, 15)

    g.addCircleArc(510, 50, 511, 50)
    g.addCircleArc(511, 50, 512, 51)
    g.addCircleArc(512, 50, 520, 52)
    g.addCircleArc(520, 50, 510, 53)
    g.addCurveLoop([50, 51, 52, 53], 99)
    g.addCurveLoop([101, 102, 103, 104, 105, 106, 107, 108,
                    109, 110, 111, 112, 113, 114, 115, 116],
                   100)

    g.addPlaneSurface([99, 100], 100)
    g.addPlaneSurface([99], 99)
    # g.copy([(2, 98))

    # we create the other segments
    g.addLine(310, 210, 117)
    g.addCurveLoop([103, 104, 105, 106, 107, 117], 101)
    g.addPlaneSurface([101], 101)
    g.addLine(110, 410, 118)
    g.addCurveLoop([111, 112, 113, 114, 115, 118], 102)
    g.addPlaneSurface([102], 102)
    # gmsh.model.addPhysicalGroup(2, [99, 101, 102], 5, "Holes")
    g.synchronize()


def AlO3(Lx, Ly, t3, w_g, lc, c):
    g = gmsh.model.geo
    w_g2 = (Lx-w_g)/2
    g.addPoint(c[0] + w_g2, c[1], c[2], lc)
    g.addPoint(c[0] + w_g2 + w_g, c[1], c[2], lc)
    g.addPoint(c[0]+Lx, c[1], c[2], lc)
    g.addPoint(c[0]+Lx, c[1]+Ly, c[2], lc)
    g.addPoint(c[0] + w_g2 + w_g, c[1]+Ly, c[2], lc)
    g.addPoint(c[0] + w_g2, c[1]+Ly, c[2], lc)
    g.addPoint(c[0], c[1]+Ly, c[2], lc)  # 528

    #  lines of top surface (120-129)
    g.addLine(521, 522, 120)
    g.addLine(522, 523)
    g.addLine(523, 524)
    g.addLine(524, 525)
    g.addLine(525, 526)
    g.addLine(526, 527)
    g.addLine(527, 528)
    g.addLine(528, 521)  # 127
    g.addLine(527, 522)
    g.addLine(523, 526)  

    # vertical lines from pg to top surface (25-28)

    g.addLine(15, 521, 25)
    g.addLine(16, 524, 26)
    g.addLine(17, 525, 27)
    g.addLine(18, 528, 28)

    # lateral surfaces
    g.addCurveLoop([101, 26, -122, -121, -120, -25])  # 103
    g.addCurveLoop([102, -117, 108, 27, -123, -26])
    g.addCurveLoop([109, 28, -126, -125, -124, -27])
    g.addCurveLoop([-116, 118, -110, 28, 127, -25])
    g.addPlaneSurface([103], 103)
    g.addPlaneSurface([104], 104)
    g.addPlaneSurface([105], 105)
    g.addPlaneSurface([106], 106)

    # top surfaces
    g.addCurveLoop([120, -128, 126, 127])  # 107
    g.addCurveLoop([122, 123, 124, -129])
    g.addCurveLoop([121, 129, 125, 128])  # gold gate, 109
    g.addPlaneSurface([107], 107)
    g.addPlaneSurface([108], 108)
    g.addPlaneSurface([109], 109)

    g.addSurfaceLoop([99, 100, 101, 102, 103, 104,
                      105, 106, 107, 108, 109])  # 1001
    g.addVolume([1001])


def create_geometry_mesh(geom_params, lc, c, filename, create_pg=True, create_gg=True):
    """
    Args:
        geometry: np.array containing:
            -Lx, Ly: float defining the dimensions on x and y of the device.
            -t1, t2: floats defining the height of the physical volumes 1
            (from the gate to the bilayer graphene)
            and 2 (from the bilayer graphene to the patterned gate)
            -create_pg

    Returns:
        -.gmsh containing the mesh with the name "filename".
    """
    Lx, Ly, t1, t2, t3, w, r, d, h, diam, w_g = unpack_geom(geom_params)
    gmsh.initialize()
    gmsh.model.add("device1")
    g = gmsh.model.geo
    g.addPoint(c[0], c[1], 0, lc, 1)
    g.addPoint(c[0]+Lx, c[1], 0, lc, 2)
    g.addPoint(c[0]+Lx, c[1]+Ly, 0, lc, 3)
    g.addPoint(c[0], c[1]+Ly, 0, lc, 4)

    g.addLine(1, 2, 1)
    g.addLine(2, 3, 2)
    g.addLine(3, 4, 3)
    g.addLine(4, 1, 4)

    g.addCurveLoop([1, 2, 3, 4], 1)
    g.addPlaneSurface([1], 1)  # Backdoor gate surface

    ex1 = g.extrude([(2, 1)], 0, 0, t1)  # extrude
    g.synchronize()

    # The points 5, 6, 10 and 14 the one generared by the extrude.

    xyz = gmsh.model.getValue(0, 5, [])
    g.addPoint(xyz[0], xyz[1], xyz[2] + t2, lc, 15)
    g.addPoint(xyz[0]+Lx, xyz[1],  xyz[2] + t2, lc, 16)
    g.addPoint(xyz[0]+Lx, xyz[1]+Ly,  xyz[2] + t2, lc, 17)
    g.addPoint(xyz[0], xyz[1]+Ly,  xyz[2] + t2, lc, 18)

    g.addLine(5, 15, 21)
    g.addLine(6, 16)
    g.addLine(10, 17)
    g.addLine(14, 18)
    g.synchronize()
    gmsh.model.addPhysicalGroup(3, [ex1[1][1]], 1, "Volume 1")
    gmsh.model.addPhysicalGroup(2, [1], 1, "Back gate")
    if create_pg:

        xyz15 = gmsh.model.getValue(0, 15, [])
        patt_gate(Lx, Ly, w, r, d, h, diam, lc, c=xyz15[0:3])

        g.addCurveLoop([6, 22, -101, -21], 50)
        g.addPlaneSurface([50], 50)
        g.addCurveLoop([22, 102, -117, 108, -23, -7], 51)
        g.addPlaneSurface([51], 51)
        g.addCurveLoop([8, 24, -109, -23], 52)
        g.addPlaneSurface([52], 52)
        g.addCurveLoop([24, 110, -118, 116, -21, -9], 53)
        g.addPlaneSurface([53], 53)
        g.addSurfaceLoop([ex1[0][1], 50, 51, 52, 53, 99, 100, 101, 102], 1000)
        g.addVolume([1000], 1000)

        # g.mesh.setSize([(0, 5), (0, 6), (0, 10), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18)], lc)
        g.synchronize()
        
        gmsh.model.addPhysicalGroup(2, [100], 2, "Patterned gate")
        gmsh.model.addPhysicalGroup(3, [1000], 2, "Volume 2")
        
        if create_gg:
            # we now create the Al3O2 volume, and the gold surface.
            # this has width t3 and the gold w_g

            # points from the top surface (521-528)
            g.translate(g.copy([(0, 15)]), 0, 0, t3)  # is the point 521
            g.synchronize()
            xyz521 = gmsh.model.getValue(0, 521, [])

            AlO3(Lx, Ly, t3, w_g, lc, xyz521)

            g.synchronize()
            gmsh.model.addPhysicalGroup(3, [1001], 3, "Volume 3 ")
            gmsh.model.addPhysicalGroup(2, [109], 3, "Gold Gate")
    g.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(filename)
    gmsh.finalize()

def unpack_geom(g):
    # geom = [Lx, Ly, t1, t2, t3, w, r, d, h, diam, w_g]
    geom = [None, None, None, None, None, None, None, None, None, None, None]
    if 'Lx' in g:
        geom[0] = g['Lx']
    if 'Ly' in g:
        geom[1] = g['Ly']
    if 't1' in g:
        geom[2] = g['t1']
    if 't2' in g:
        geom[3] = g['t2']
    if 't3' in g:
        geom[4] = g['t3']
    if 'w' in g:
        geom[5] = g['w']
    if 'r' in g:
        geom[6] = g['r']
    if 'd' in g:
        geom[7] = g['d']
    if 'h' in g:
        geom[8] = g['h']
    if 'diam' in g:
        geom[9] = g['diam']
    if 'w_g' in g:
        geom[10] = g['w_g']
    return geom

