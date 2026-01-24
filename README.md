# Hovercraft Simulation

A simple reinforcement learning environment simulation for a hovercraft using Open3D for visualization.

## Current Status

✅ **Implemented Features:**
- Complete hovercraft physics simulation with Gaussian forces
- Rectangular fence boundary with bouncing mechanics
- Mass-based Newtonian physics (F=ma, torque responses)
- Friction proportional to elevation
- Controlled forward thrust and rotational torque
- Test suite demonstrating all physics behaviors

⚠️ **Known Issues:**
- Open3D visualization not available for Python 3.13 (requires Python ≤3.12)
- Physics simulation runs correctly without visualization

## Installation

This project uses modern Python packaging with `pyproject.toml`.

### Prerequisites
- Python >= 3.13 (physics simulation works)
- Python <= 3.12 (required for Open3D visualization)

### Install Dependencies
```bash
pip install -e .
```

This installs numpy for the physics simulation.

**Note:** Open3D is not included due to Python 3.13 compatibility. Visualization will fail until Open3D supports Python 3.13.

## Usage

### Testing Physics (Works Now)
Run the demonstration tests to verify physics:
```bash
python test_demo.py
```

This runs tests for:
- Hovering behavior with random forces
- Forward movement and boundary bouncing
- Rotational control
- Combined physics simulation

### Visualization (Future)
Once Open3D supports Python 3.13:
```bash
python main.py
```

This will open an Open3D visualization window showing the hovercraft moving within the fenced training area.

## Future Steps

### Immediate (High Priority)
1. **Resolve Visualization Compatibility**
   - Monitor Open3D releases for Python 3.13 support
   - OR implement alternative 3D visualization (e.g., PyVista, VTK, or WebGL-based)
   - OR downgrade to Python 3.12 for full functionality

2. **Add Reinforcement Learning Interface**
   - Implement Gymnasium-compatible environment
   - Add reward functions for RL training
   - Define observation and action spaces

### Medium Term
3. **Enhanced Physics**
   - Add more realistic aerodynamic effects
   - Implement collision detection with obstacles
   - Add wind disturbances

4. **Performance Optimization**
   - Optimize visualization rendering
   - Add parallel simulation capabilities

### Long Term
5. **Advanced Features**
   - Multi-agent scenarios
   - Terrain interaction
   - Sensor simulation (lidar, camera)

## Contributing

The physics simulation is complete and tested. Contributions welcome for:
- Alternative visualization implementations
- RL integration
- Additional physics features
- Performance improvements

## License

Add your license here