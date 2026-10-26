"""Native sheet meshes preserve ground holes and expose edge grading."""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import gmsh
import numpy as np
import pytest
from pydantic import ValidationError

from gsim.common.stack import Layer, LayerStack
from gsim.palace import ElectrostaticSim
from gsim.palace.models import MeshConfig


def sheet_sim(path: Path) -> ElectrostaticSim:
    """Two electrodes inside a grounded frame on a dielectric interface."""
    gf.gpdk.PDK.activate()
    component = gf.Component()
    for name, y in [("lower", -4), ("upper", 4)]:
        component.add_polygon(
            [(0, y - 1), (4, y - 1), (4, y + 1), (0, y + 1)], layer=(1, 0)
        )
        component.add_port(
            name=name, center=(0, y), width=2, orientation=180,
            layer=(1, 0), port_type="electrical",
        )
    inner = gf.Component()
    inner.add_ref(gf.c.rectangle(size=(10, 16), layer=(1, 0))).dmove((2, 17))
    frame = gf.boolean(
        gf.c.rectangle(size=(14, 50), layer=(1, 0)), inner,
        operation="not", layer=(1, 0),
    )
    component.add_ref(frame).dmove((-5, -25))
    stack = LayerStack(
        layers={"metal": Layer(
            name="metal", gds_layer=(1, 0), zmin=0, zmax=0, thickness=0,
            material="aluminum", layer_type="conductor",
        )},
        materials={
            "aluminum": {"type": "conductor", "conductivity": 3.77e7},
            "silicon": {"type": "dielectric", "permittivity": 11.45},
            "air": {"type": "dielectric", "permittivity": 1},
        },
        dielectrics=[{"name": "silicon", "zmin": -25, "zmax": 0, "material": "silicon"}],
    )
    sim = ElectrostaticSim()
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.set_airbox(margin_x=0, margin_y=0, z_above=25, z_below=0)
    sim.set_output_dir(path)
    sim.add_terminal("T1", net="lower")
    sim.add_terminal("T2", net="upper")
    sim.set_unassigned_conductors("ground")
    return sim


def test_sheet_ground_hole_and_configured_edge_field(tmp_path: Path, monkeypatch) -> None:
    sim = sheet_sim(tmp_path)
    sim.mesh_config = MeshConfig(
        refinement_distance_min=1, refinement_distance_max=8, refinement_sampling=1000
    )
    fields = {}
    generate = gmsh.model.mesh.generate

    def record_fields(dim: int) -> None:
        for tag in gmsh.model.mesh.field.list():
            kind = gmsh.model.mesh.field.getType(tag)
            if kind == "Threshold":
                fields[kind] = {
                    name: gmsh.model.mesh.field.getNumber(tag, name)
                    for name in ("DistMin", "DistMax", "SizeMin", "SizeMax")
                }
            elif kind == "Distance":
                fields[kind] = gmsh.model.mesh.field.getNumber(tag, "Sampling")
        generate(dim)

    monkeypatch.setattr(gmsh.model.mesh, "generate", record_fields)
    mesh = sim.mesh(planar_conductors=True, refined_mesh_size=0.5, max_mesh_size=5, verbose=False)
    config = json.loads(sim.write_config().read_text())
    assert sim.validate_mesh().valid
    assert fields["Threshold"] == {"DistMin": 1, "DistMax": 8, "SizeMin": 0.5, "SizeMax": 5}
    assert fields["Distance"] == 1000
    terminals = [set(t["Attributes"]) for t in config["Boundaries"]["Terminal"]]
    ground = set(config["Boundaries"]["Ground"]["Attributes"])
    assert len(terminals) == 2 and all(terminals) and ground
    assert terminals[0].isdisjoint(terminals[1])
    assert ground.isdisjoint(terminals[0] | terminals[1])
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(mesh.mesh_path))
        tags, coords, _ = gmsh.model.mesh.getNodes()
        points = dict(zip(tags, np.asarray(coords).reshape(-1, 3), strict=True))
        areas = {}
        for dim, group in gmsh.model.getPhysicalGroups(2):
            name = gmsh.model.getPhysicalName(dim, group)
            if not name.startswith("metal_pec@"):
                continue
            area = 0.0
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, group):
                _, nodes = gmsh.model.mesh.getElementsByType(2, int(entity))
                for triangle in np.asarray(nodes).reshape(-1, 3):
                    a, b, c = (points[tag] for tag in triangle)
                    area += np.linalg.norm(np.cross(b - a, c - a)) / 2
            areas[name] = area
        assert sorted(areas.values()) == pytest.approx([8, 8, 540])
    finally:
        gmsh.finalize()


@pytest.mark.parametrize("kwargs", [
    {"refinement_distance_min": -1},
    {"refinement_distance_min": float("nan")},
    {"refinement_distance_max": float("inf")},
    {"refinement_distance_max": 0},
    {"refinement_distance_min": 2, "refinement_distance_max": 2},
    {"refinement_sampling": 1},
])
def test_invalid_edge_controls(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        MeshConfig(**kwargs)
