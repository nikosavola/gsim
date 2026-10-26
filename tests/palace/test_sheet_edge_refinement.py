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
from gsim.palace.models import CrossSectionPlaneConfig, MeshConfig
from tests._electrode_layouts import make_sim


def sheet_sim(path: Path) -> ElectrostaticSim:
    """Two electrodes inside a grounded frame on a dielectric interface."""
    gf.gpdk.PDK.activate()
    component = gf.Component()
    for name, y in [("lower", -4), ("upper", 4)]:
        component.add_polygon(
            [(0, y - 1), (4, y - 1), (4, y + 1), (0, y + 1)], layer=(1, 0)
        )
        component.add_port(
            name=name,
            center=(0, y),
            width=2,
            orientation=180,
            layer=(1, 0),
            port_type="electrical",
        )
    inner = gf.Component()
    inner.add_ref(gf.c.rectangle(size=(10, 16), layer=(1, 0))).dmove((2, 17))
    frame = gf.boolean(
        gf.c.rectangle(size=(14, 50), layer=(1, 0)),
        inner,
        operation="not",
        layer=(1, 0),
    )
    component.add_ref(frame).dmove((-5, -25))
    stack = LayerStack(
        layers={
            "metal": Layer(
                name="metal",
                gds_layer=(1, 0),
                zmin=0,
                zmax=0,
                thickness=0,
                material="aluminum",
                layer_type="conductor",
            )
        },
        materials={
            "aluminum": {"type": "conductor", "conductivity": 3.77e7},
            "silicon": {"type": "dielectric", "permittivity": 11.45},
            "air": {"type": "dielectric", "permittivity": 1},
        },
        dielectrics=[
            {"name": "silicon", "zmin": -25, "zmax": 0, "material": "silicon"}
        ],
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


@pytest.mark.parametrize("refine_bulk", [False, True])
def test_sheet_ground_hole_and_configured_edge_field(
    tmp_path: Path, monkeypatch, refine_bulk: bool
) -> None:
    sim = sheet_sim(tmp_path)
    sim.mesh_config = MeshConfig(
        refinement_distance_min=1,
        refinement_distance_max=8,
        refinement_sampling=1000,
        refine_dielectric_bulk=refine_bulk,
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
            elif kind == "Box":
                fields[kind] = gmsh.model.mesh.field.getNumber(tag, "VIn")
        generate(dim)

    monkeypatch.setattr(gmsh.model.mesh, "generate", record_fields)
    mesh = sim.mesh(
        planar_conductors=True, refined_mesh_size=0.5, max_mesh_size=5, verbose=False
    )
    config = json.loads(sim.write_config().read_text())
    assert sim.validate_mesh().valid
    assert fields["Threshold"] == {
        "DistMin": 1,
        "DistMax": 8,
        "SizeMin": 0.5,
        "SizeMax": 5,
    }
    assert fields["Distance"] == 1000
    assert ("Box" in fields) is refine_bulk
    terminals = [set(t["Attributes"]) for t in config["Boundaries"]["Terminal"]]
    ground = set(config["Boundaries"]["Ground"]["Attributes"])
    assert len(terminals) == 2
    assert all(terminals)
    assert ground
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"refinement_distance_min": -1},
        {"refinement_distance_min": float("nan")},
        {"refinement_distance_max": float("inf")},
        {"refinement_distance_max": 0},
        {"refinement_distance_min": 2, "refinement_distance_max": 2},
        {"refinement_sampling": 1},
    ],
)
def test_invalid_edge_controls(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        MeshConfig(**kwargs)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_native_electrostatic_section_keeps_layout_connectivity(
    tmp_path: Path, axis: str
) -> None:
    sim = sheet_sim(tmp_path)
    if axis == "y":
        rotated = gf.Component()
        reference = rotated.add_ref(sim.component)
        reference.drotate(90)
        rotated.add_ports(reference.ports)
        sim.set_geometry(rotated)
    sim.set_cross_section(f"{axis}=2", normalization_depth_um=4)
    result = sim.mesh(
        planar_conductors=True, refined_mesh_size=0.5, max_mesh_size=5, verbose=False
    )
    config = json.loads(sim.write_config().read_text())
    assert config["Model"]["Lc"] == 4
    assert result.mesh_stats.get("tetrahedra", 0) == 0
    assert sim.validate_mesh().valid
    assert len(config["Boundaries"]["Terminal"]) == 2
    assert len(config["Boundaries"]["Ground"]["Attributes"]) == 1
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(result.mesh_path))
        _, edges = gmsh.model.mesh.getElementsByType(1)
        edge_nodes = np.sort(np.asarray(edges).reshape(-1, 2), axis=1)
        assert len(np.unique(edge_nodes, axis=0)) == len(edge_nodes)
        lengths = {}
        for dim, group in gmsh.model.getPhysicalGroups(1):
            name = gmsh.model.getPhysicalName(dim, group)
            if not name.startswith("metal@"):
                continue
            total = 0.0
            for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, group):
                _, coords, _ = gmsh.model.mesh.getNodes(
                    1, int(entity), includeBoundary=True
                )
                points = np.asarray(coords).reshape(-1, 3)
                total += points[:, 0].max() - points[:, 0].min()
                element_tags, _ = gmsh.model.mesh.getElementsByType(1, int(entity))
                assert len(element_tags) >= 4
            lengths[name] = total
        assert sorted(lengths.values()) == pytest.approx([2, 2, 34])
    finally:
        gmsh.finalize()


def test_section_rejects_a_terminal_outside_the_window(tmp_path: Path) -> None:
    sim = sheet_sim(tmp_path)
    sim.set_cross_section("x=2", window=(-6, 0))
    sim.mesh(
        planar_conductors=True, refined_mesh_size=1, max_mesh_size=5, verbose=False
    )
    with pytest.raises(ValueError, match=r"T2.*no surfaces"):
        sim.write_config()


@pytest.mark.parametrize("depth", [0, -1, float("inf"), float("nan")])
def test_section_depth_must_be_positive_and_finite(
    tmp_path: Path, depth: float
) -> None:
    with pytest.raises(ValidationError):
        sheet_sim(tmp_path).set_cross_section("x=2", normalization_depth_um=depth)


def test_section_rejects_an_xy_cut(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="x or y"):
        sheet_sim(tmp_path).set_cross_section("z=0")


def test_section_resolves_a_ground_strip_smaller_than_the_mesh_target(
    tmp_path: Path,
) -> None:
    sim = sheet_sim(tmp_path)
    component = sim.component
    assert component is not None
    component.add_polygon(
        [(0, -0.006), (4, -0.006), (4, 0.006), (0, 0.006)], layer=(1, 0)
    )
    sim.set_cross_section("x=2")
    result = sim.mesh(
        planar_conductors=True, refined_mesh_size=0.5, max_mesh_size=5, verbose=False
    )
    config = json.loads(sim.write_config().read_text())
    found = False
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(result.mesh_path))
        for group in config["Boundaries"]["Ground"]["Attributes"]:
            for entity in gmsh.model.getEntitiesForPhysicalGroup(1, group):
                _, coords, _ = gmsh.model.mesh.getNodes(
                    1, int(entity), includeBoundary=True
                )
                points = np.asarray(coords).reshape(-1, 3)
                length = points[:, 0].max() - points[:, 0].min()
                if length < 0.02:
                    elements, _ = gmsh.model.mesh.getElementsByType(1, int(entity))
                    assert length == pytest.approx(0.012)
                    assert len(elements) >= 4
                    found = True
        assert found
    finally:
        gmsh.finalize()


def test_section_preserves_layer_only_terminal_selection(tmp_path: Path) -> None:
    sim = sheet_sim(tmp_path)
    component, stack = sim.component, sim.stack
    assert component is not None
    assert stack is not None
    component.add_polygon([(0, -1), (4, -1), (4, 1), (0, 1)], layer=(2, 0))
    stack.layers["metal2"] = Layer(
        name="metal2",
        gds_layer=(2, 0),
        zmin=2,
        zmax=2,
        thickness=0,
        material="aluminum",
        layer_type="conductor",
    )
    sim.terminals.clear()
    sim.add_terminal("LOWER", layer="metal")
    sim.add_terminal("UPPER", layer="metal2")
    sim.set_cross_section("x=2")
    sim.mesh(refined_mesh_size=1, max_mesh_size=5, verbose=False)
    config = json.loads(sim.write_config().read_text())
    terminals = config["Boundaries"]["Terminal"]
    assert len(terminals) == 2
    assert all(terminal["Attributes"] for terminal in terminals)
    assert set(terminals[0]["Attributes"]).isdisjoint(terminals[1]["Attributes"])
    assert "Ground" not in config["Boundaries"]


@pytest.mark.parametrize(
    ("planar", "message"), [(False, "zero-thickness"), (True, "vias")]
)
def test_section_rejects_unsupported_stack_geometry(
    tmp_path: Path, planar: bool, message: str
) -> None:
    sim = make_sim(tmp_path)
    sim.add_terminal("T1", net="PLUS")
    sim.add_terminal("T2", net="MINUS")
    sim.set_cross_section("x=2")
    with pytest.raises(ValueError, match=message):
        sim.mesh(planar_conductors=planar, verbose=False)


def test_section_rejects_periodic_meshing(tmp_path: Path) -> None:
    sim = sheet_sim(tmp_path)
    sim.set_cross_section("x=2")
    with pytest.raises(ValueError, match="periodic"):
        sim.mesh(periodic_axis="x", verbose=False)


def test_section_rejects_an_xy_plane_config_when_meshing(tmp_path: Path) -> None:
    sim = sheet_sim(tmp_path)
    sim.cross_section = CrossSectionPlaneConfig(axis="z", value=0)
    with pytest.raises(ValueError, match="only x/y"):
        sim.mesh(verbose=False)
