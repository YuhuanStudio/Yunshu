"""Yunshu Control Plane — Mesh management router.

Node discovery, topology management, distributed ops monitoring.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/mesh", tags=["mesh"])


def _get_mesh_manager(request: Request):
    from yunshu_mesh.manager import MeshManager

    manager = getattr(request.app.state, "mesh_manager", None)
    if manager is None:
        manager = MeshManager()
        request.app.state.mesh_manager = manager
    return manager


@router.get("/status")
async def mesh_status(request: Request):
    """Get overall mesh status."""
    manager = _get_mesh_manager(request)
    return manager.get_stats()


@router.get("/nodes")
async def list_nodes(request: Request):
    """List all nodes in the mesh."""
    manager = _get_mesh_manager(request)
    topo = manager.topology
    return {
        "nodes": [n.to_dict() for n in topo.nodes],
        "total": len(topo.nodes),
        "topology_type": topo.topo_type.value,
    }


@router.post("/initialize")
async def initialize_mesh(request: Request, backend: str = "any", topology: str | None = None):
    """Initialize the mesh with a specific backend and topology."""
    from yunshu_mesh.topology import TopologyType

    manager = _get_mesh_manager(request)

    topo_type = None
    if topology:
        try:
            topo_type = TopologyType(topology)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid topology: {topology}")

    success = manager.initialize(backend=backend, topology_type=topo_type)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to initialize mesh")

    return {"status": "initialized", **manager.get_stats()}


@router.post("/pipeline/setup")
async def setup_pipeline(request: Request, num_layers: int, memory_per_layer_gb: float = 1.0):
    """Setup pipeline parallelism for a model."""
    manager = _get_mesh_manager(request)
    pipeline = manager.setup_pipeline(num_layers, memory_per_layer_gb)
    return pipeline.to_dict()


@router.get("/collective/test")
async def test_collective(request: Request):
    """Test collective operations (all_reduce, all_gather)."""
    manager = _get_mesh_manager(request)

    if not manager.is_distributed:
        return {"status": "single_node", "message": "No distributed backend initialized"}

    results = manager.collective.run_benchmark()
    return {"status": "ok", "results": results}
