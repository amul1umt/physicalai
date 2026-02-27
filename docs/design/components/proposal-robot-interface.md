# Robot Protocol Interface Design

## Overview

The robot interface defines how `physicalai` communicates with physical robots during inference deployment. It is deliberately minimal — the robot is plumbing in service of the inference loop, not the product itself.

The interface uses Python's `Protocol` for structural typing. Robot implementations do not inherit from a base class. They implement the required methods, and duck typing handles the rest.

## The TLDR

### The Interface

Four methods. No base class.

```python
class Robot(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_observation(self) -> dict[str, Any]: ...
    def send_action(self, action: np.ndarray) -> None: ...
```

Any class that implements these four methods is a valid robot. No inheritance, no registration, no dependency on `physicalai`.

### Protocol Over ABC

| Concern | ABC | Protocol                                                        |
|---|---|-----------------------------------------------------------------|
| Third-party robots | Must `from physicalai.robots import Robot` and subclass | Just implement the methods. No import needed.                   |
| Multiple robotics libraries | MRO conflicts if two libraries define a `Robot` ABC | No inheritance, no conflicts                                    |
| Testing | Must subclass to create a mock | Mocks work directly                                             |
| Adding optional capabilities later | Add abstract method → breaks all implementations | Add a separate Protocol → existing code untouched               |
| Error on missing method | Immediate at instantiation | At first call (mitigated by conformance tests and type checkers) |

### Usage

```python
from physicalai.inference import InferenceModel
from physicalai.robots import managed
from robots import SO101

model = InferenceModel("./my_policy")
robot = SO101(port="/dev/ttyUSB0")

with managed(robot):
    obs = robot.get_observation()
    action = model(obs)
    robot.send_action(action)
```

## Design Principles

- **Protocol, not inheritance.** No base class to subclass. Implementations are plain classes that happen to have the right methods.
- **Standard Python types.** Observations are `dict[str, Any]`, actions are `np.ndarray`. No custom message types, no protobuf, no ROS messages.
- **Synchronous.** All methods are blocking. If an implementation needs async I/O internally (e.g., for cameras), it bridges to sync at the boundary.
- **Validation via manifest.** The robot does not describe itself. The policy's `manifest.json` declares what it expects. The runtime validates observations against the manifest on first contact.
- **Safe disconnect.** `disconnect()` must leave the robot in a safe, stationary state. This is a contractual requirement on every implementation.

## Why Protocol, Not ABC

The standard Python approach for defining an interface is an Abstract Base Class (ABC). We use `Protocol` instead. This section explains why.

### Zero coupling for third-party implementations

With an ABC, every robot implementation must import and inherit from `physicalai`:

```python
# ABC approach — third party MUST depend on physicalai
from physicalai.robots import Robot

class MyRobot(Robot):
    def connect(self): ...
    def disconnect(self): ...
    def get_observation(self): ...
    def send_action(self, action): ...
```

With Protocol, the implementation has no dependency on `physicalai` at all:

```python
# Protocol approach — no import, no inheritance
class MyRobot:
    def connect(self): ...
    def disconnect(self): ...
    def get_observation(self): ...
    def send_action(self, action): ...
```

If a third party already has a working robot driver with the right methods, it works as-is. No adapter, no wrapper, no added dependency.

### No hierarchy conflicts

In the robotics ecosystem, users combine multiple libraries — LeRobot, ROS, Gymnasium, custom drivers. If two libraries both define a `Robot` ABC, a class can't cleanly inherit from both without MRO (Method Resolution Order) conflicts. Protocols have no inheritance, so there are no conflicts. A single class can satisfy multiple Protocols from different libraries simultaneously.

### Simpler testing

With an ABC, creating a test double requires subclassing:

```python
# ABC — must subclass to mock
class FakeRobot(Robot):
    def connect(self): pass
    def disconnect(self): pass
    def get_observation(self):
        return {"state": np.zeros(6), "timestamp": 0.0}
    def send_action(self, action): pass
```

With Protocol, standard mocking tools work directly:

```python
# Protocol — Mock satisfies the interface
from unittest.mock import Mock

robot = Mock()
robot.get_observation.return_value = {"state": np.zeros(6), "timestamp": 0.0}
run_episode(robot, policy)  # works
```

### Forward compatibility

If a new optional capability is needed later (e.g., camera intrinsics), with an ABC the choices are bad:
- Add it as abstract → breaks all existing implementations.
- Add it with a default → implementations silently inherit behavior they may not want.

With Protocol, define a separate Protocol for the new capability:

```python
class SupportsIntrinsics(Protocol):
    def get_camera_intrinsics(self, name: str) -> np.ndarray: ...
```

Functions that need it accept the extended type. Functions that don't still accept plain `Robot`. Existing implementations are untouched.

### The right stance for an inference library

`physicalai` is an inference deployment runtime, not a robot framework. The robot interface exists to feed observations into models and send actions out. The lightest possible contract with the hardware layer keeps the focus where it belongs — on inference.

An ABC says: "you **are** a Robot." A Protocol says: "you **can do** what we need." The latter is the appropriate relationship between an inference runtime and the hardware it talks to.

### The trade-off

The cost of Protocol over ABC: no `TypeError` at class definition time if a method is missing. With an ABC, forgetting `send_action()` raises an error the moment the class is instantiated. With Protocol, the error surfaces later when the runtime calls the missing method.

This is acceptable because:
- The conformance test suite (`check_robot_conformance()`) catches missing methods immediately when run.
- Contributors test their robot implementation within seconds of writing it — a robot driver is not something you write and leave untested.
- Static type checkers (`mypy`, `pyright`) flag Protocol violations before runtime if type annotations are used.

## Protocol Definition

```python
# physicalai/robots/protocol.py
from typing import Any, Protocol
import numpy as np


class Robot(Protocol):
    """Structural interface for robot implementations.

    Any class that implements these four methods is a valid robot.
    No inheritance required. No registration required.
    """

    def connect(self) -> None:
        """Establish connection to the robot hardware.

        Called once before the inference loop begins. Must be idempotent —
        calling connect() on an already-connected robot should be a no-op
        or raise a clear error.
        """
        ...

    def disconnect(self) -> None:
        """Disconnect from the robot.

        Implementations MUST leave the robot in a safe, stationary state.
        Motors must be stopped or holding position before the connection
        is closed. This method is called automatically by the managed()
        context manager, including when exceptions occur.
        """
        ...

    def get_observation(self) -> dict[str, Any]:
        """Read the current robot state and sensor data.

        Returns a dict with the following conventional structure:

            {
                "images": {
                    "camera_name": np.ndarray,  # (C, H, W) uint8 or float32
                    ...
                },
                "state": np.ndarray,            # joint positions, gripper, etc.
                "timestamp": float,             # time.monotonic() or equivalent
            }

        The exact keys and shapes must match what the policy expects,
        as declared in the policy's manifest.json under io.inputs.

        Implementations that have no cameras may omit the "images" key.
        """
        ...

    def send_action(self, action: np.ndarray) -> None:
        """Send an action command to the robot.

        Args:
            action: A numpy array of joint commands. The shape and semantics
                    (positions, velocities, torques) depend on the policy
                    that produced the action. The robot implementation is
                    responsible for interpreting them correctly.
        """
        ...
```

## Context Manager

The Protocol cannot provide default method implementations. Instead, `physicalai` ships a context manager wrapper:

```python
# physicalai/robots/utils.py
from contextlib import contextmanager


@contextmanager
def managed(robot):
    """Context manager for safe robot lifecycle.

    Calls connect() on entry and disconnect() on exit, including
    when exceptions occur.

    Usage:
        with managed(robot):
            obs = robot.get_observation()
            robot.send_action(action)
    """
    robot.connect()
    try:
        yield robot
    finally:
        robot.disconnect()
```

## Implementing a Robot

Adding a new robot is straightforward. Implement the four methods:

```python
# physicalai/robots/so100.py
import time
import numpy as np


class SO100:
    """Concrete implementation for the SO-100 robot arm."""

    def __init__(self, port: str, cameras: dict | None = None):
        self.port = port
        self.cameras = cameras or {}
        self._connection = None

    def connect(self) -> None:
        self._connection = serial.Serial(self.port, baudrate=1_000_000)
        for cam in self.cameras.values():
            cam.start()

    def disconnect(self) -> None:
        # Stop all motors before closing connection
        if self._connection:
            self._write_torque(enabled=False)
            self._connection.close()
            self._connection = None
        for cam in self.cameras.values():
            cam.stop()

    def get_observation(self) -> dict[str, Any]:
        images = {
            name: cam.read().data
            for name, cam in self.cameras.items()
        }
        state = self._read_joint_positions()
        return {
            "images": images,
            "state": state,
            "timestamp": time.monotonic(),
        }

    def send_action(self, action: np.ndarray) -> None:
        self._write_joint_positions(action)
```

No base class imported. No registration. The class satisfies the `Robot` protocol by having the right methods.

### Third-Party Robots

Third-party implementations follow the same pattern. They do not need to import anything from `physicalai`:

```python
# In a user's own code or separate package

class MyCustomRobot:
    def connect(self) -> None:
        # custom hardware setup
        ...

    def disconnect(self) -> None:
        # stop motors, close connection
        ...

    def get_observation(self) -> dict[str, Any]:
        return {
            "state": np.array([...]),
            "timestamp": time.monotonic(),
        }

    def send_action(self, action: np.ndarray) -> None:
        # send commands to hardware
        ...
```

This works with the `physicalai` runtime without modification:

```python
from physicalai.inference import InferenceModel
from physicalai.robots import managed
from my_package import MyCustomRobot

model = InferenceModel("./my_policy")
robot = MyCustomRobot(port="/dev/ttyUSB0")

with managed(robot):
    obs = robot.get_observation()
    action = model(obs)
    robot.send_action(action)
```

## Multi-Arm vs Multi-Robot

**Multi-arm** (e.g., bimanual robot): A single class with wider state and action vectors. Both arms are one robot.

```python
class BimanualRobot:
    def get_observation(self) -> dict[str, Any]:
        return {
            "state": np.concatenate([left_joints, right_joints]),  # (12,)
            "timestamp": time.monotonic(),
        }

    def send_action(self, action: np.ndarray) -> None:
        left_action = action[:6]
        right_action = action[6:]
        ...
```

**Multiple independent robots**: Multiple instances, managed separately.

```python
left_robot = SO100(port="/dev/ttyUSB0")
right_robot = SO100(port="/dev/ttyUSB1")

with managed(left_robot), managed(right_robot):
    ...
```

## Cameras

Two patterns for camera integration:

### Robot-managed cameras

Cameras are passed to the robot at construction. `get_observation()` returns images alongside joint state.

```python
robot = SO100(
    port="/dev/ttyUSB0",
    cameras={"wrist": OpenCVCamera(index=0)},
)

with managed(robot):
    obs = robot.get_observation()
    # obs["images"]["wrist"] contains the camera frame
```

### User-managed cameras

User reads cameras separately and assembles the observation dict:

```python
camera = OpenCVCamera(index=0)
robot = SO100(port="/dev/ttyUSB0")

with managed(robot):
    frame = camera.read()
    obs = robot.get_observation()
    obs["images"] = {"wrist": frame.data}
```

The choice depends on the use case. Robot-managed cameras are simpler. User-managed cameras allow custom preprocessing or non-standard camera setups.

## Validation

The robot does not describe its own capabilities. Instead, the policy's `manifest.json` declares what it expects, and the runtime validates against reality.

### Manifest as source of truth

The policy manifest already contains input/output specifications:

```json
{
    "policy": {
        "kind": "ActionChunkingPolicy",
        "control_frequency_hz": 50
    },
    "io": {
        "inputs": {
            "images.wrist": { "shape": [3, 480, 640], "dtype": "uint8" },
            "state": { "shape": [6], "dtype": "float32" }
        },
        "outputs": {
            "action": { "shape": [100, 6], "dtype": "float32" }
        }
    }
}
```

### Pre-connection: inspect policy requirements

Before connecting to any hardware, a user can inspect what the policy needs:

```python
model = InferenceModel("./my_policy")

print(model.expected_inputs)
# {'images.wrist': {'shape': [3, 480, 640], 'dtype': 'uint8'},
#  'state': {'shape': [6], 'dtype': 'float32'}}

print(model.expected_outputs)
# {'action': {'shape': [100, 6], 'dtype': 'float32'}}

# User reads this and knows: "I need a 6-DOF robot with a wrist camera at 480x640"
```

### First-contact: automatic validation

On the first call to `model(obs)`, the runtime validates observation shapes against the manifest:

```python
class InferenceModel:
    def __call__(self, inputs: dict) -> dict:
        if not self._validated:
            self._validate_inputs(inputs)
            self._validated = True
        return self._run(inputs)

    def _validate_inputs(self, inputs: dict) -> None:
        for key, spec in self._manifest["io"]["inputs"].items():
            value = self._resolve_key(inputs, key)
            if value is None:
                raise IncompatibleInputError(
                    f"Policy expects input '{key}' but it was not found "
                    f"in the observation dict."
                )
            expected_shape = tuple(spec["shape"])
            if value.shape != expected_shape:
                raise IncompatibleInputError(
                    f"Policy expects '{key}' with shape {expected_shape} "
                    f"but got {value.shape}. "
                    f"Check that the robot matches the policy's training setup."
                )

    def _resolve_key(self, inputs: dict, dotted_key: str):
        """Resolve 'images.wrist' to inputs['images']['wrist']."""
        obj = inputs
        for part in dotted_key.split("."):
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        return obj
```

This validates at the boundary between robot and policy — the natural place where mismatches surface. No robot descriptor needed. No configuration files to maintain. The manifest is the single source of truth, and reality is checked against it.

## Frequency Control

The runtime episode loop owns frequency control, not the robot. The target frequency comes from the policy manifest:

```python
# Inside physicalai.runtime episode loop
target_dt = 1.0 / manifest["policy"]["control_frequency_hz"]

while running:
    t_start = time.monotonic()

    obs = robot.get_observation()
    action = model(obs)
    robot.send_action(action)

    elapsed = time.monotonic() - t_start
    sleep_time = target_dt - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)
    else:
        log.warning(
            f"Loop overrun: {elapsed:.3f}s > {target_dt:.3f}s. "
            f"Inference may be too slow for the target frequency."
        )
```

When inference is slower than the target frequency, action chunking runners can compensate by executing multiple pre-computed actions between inference calls. This is handled at the runner level, not the robot level.

## Async and Concurrency

The Protocol is synchronous. All methods block until complete.

This is a deliberate choice. A robot control loop at 10-50Hz with 1-3 I/O sources does not benefit from `asyncio`. If an implementation needs internal concurrency (e.g., reading multiple cameras in parallel), it uses threads inside its own methods:

```python
class MultiCameraRobot:
    def get_observation(self) -> dict[str, Any]:
        with ThreadPoolExecutor() as pool:
            futures = {
                name: pool.submit(cam.read)
                for name, cam in self.cameras.items()
            }
            images = {
                name: f.result().data
                for name, f in futures.items()
            }
        return {
            "images": images,
            "state": self._read_joints(),
            "timestamp": time.monotonic(),
        }
```

The sync Protocol enforces this at the type-checking level. An `async def get_observation()` has a different return type (`Coroutine`) and will be flagged by `mypy` as incompatible with the Protocol.

## Conformance Testing

`physicalai` ships a test utility that robot implementers can run against their implementation:

```python
# physicalai/robots/testing.py
import time
import numpy as np


def check_robot_conformance(robot, num_steps: int = 10):
    """Verify a robot implementation satisfies the Protocol contract.

    Checks:
    - connect/disconnect lifecycle
    - get_observation returns the expected dict structure
    - send_action accepts a numpy array
    - disconnect leaves the robot stationary
    """
    # Lifecycle
    robot.connect()

    # Observation structure
    obs = robot.get_observation()
    assert isinstance(obs, dict), "get_observation() must return a dict"
    assert "state" in obs, "observation must contain 'state'"
    assert isinstance(obs["state"], np.ndarray), "state must be np.ndarray"
    assert "timestamp" in obs, "observation must contain 'timestamp'"
    assert isinstance(obs["timestamp"], (int, float)), "timestamp must be numeric"

    if "images" in obs:
        assert isinstance(obs["images"], dict), "images must be a dict"
        for name, img in obs["images"].items():
            assert isinstance(img, np.ndarray), f"image '{name}' must be np.ndarray"
            assert img.ndim == 3, f"image '{name}' must be 3D (C, H, W)"

    # Action
    state_dim = obs["state"].shape[0]
    action = np.zeros(state_dim, dtype=np.float32)
    robot.send_action(action)

    # Safe disconnect
    robot.disconnect()
    robot.connect()
    obs1 = robot.get_observation()
    time.sleep(0.1)
    obs2 = robot.get_observation()
    assert np.allclose(obs1["state"], obs2["state"], atol=0.01), (
        "Robot must be stationary after disconnect(). "
        f"State changed from {obs1['state']} to {obs2['state']}"
    )
    robot.disconnect()

    print("All conformance checks passed.")
```

## Summary

| Decision | Choice |
|---|---|
| Interface mechanism | `Protocol` (structural typing, no inheritance) |
| Data types | `dict[str, Any]` for observations, `np.ndarray` for actions |
| Context manager | `managed()` wrapper function |
| Safety | `disconnect()` must leave robot stationary (documented contract, conformance test) |
| Concurrency | Synchronous protocol, threads allowed internally |
| Validation | Policy manifest is source of truth, validated on first observation |
| Frequency control | Runtime episode loop, target from manifest |
| Built-in robots | `physicalai` ships concrete implementations for supported hardware |
| Third-party robots | Implement the four methods, no imports from `physicalai` required |