# Fencing drill remote

This is the small GPIO button client for a Raspberry Pi Zero 2 W. It installs
GPIO Zero only; it does not install LeRobot, PyTorch, or robot drivers.

## Install on the Pi Zero 2 W

Copy this `remote` directory to the remote Pi, then install:

```bash
sudo apt update
sudo apt install python3-lgpio python3-venv
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install .
```

Connect each normally-open momentary button between its BCM GPIO pin and a Pi
ground pin:

| Button | BCM GPIO | Keyboard command |
| --- | ---: | --- |
| En Garde / standby only | 17 | `g` |
| Start a repetition | 24 | `s` |
| Replay/reset | 27 | `r` |
| End | 22 | `q` |

The inputs use internal pull-ups. Use only Pi 3.3 V GPIO and ground; do not
connect a GPIO pin to 5 V. Override the defaults with `--enguard-pin`,
`--start-pin`, `--replay-pin`, or `--end-pin`.

The En Garde button moves the arm to the saved standby move and then waits; it
does not start the drill. The Start button begins one repetition.

## Run the robot session on the Pi 5

Generate a shared token on the robot Pi:

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

Keep the token private and run the session with the remote listener enabled:

```bash
uv run --extra feetech --extra roboclaw lerobot-fencing-robot-session \
  --config_path=examples/fencing_robot/retreat_parry3_session.json \
  --robot.arm.id=fencing_arm_01 \
  --remote=true \
  --remote_host=0.0.0.0 \
  --remote_port=8765 \
  --remote_token="PASTE_THE_SHARED_TOKEN_HERE"
```

On the Pi Zero, use the same token and the Pi 5's private Wi-Fi address:

```bash
. .venv/bin/activate
fencing-remote --host=192.168.1.50 --token="PASTE_THE_SHARED_TOKEN_HERE"
```

Press **En Garde / standby only** first to verify that the remote is received
and the arm moves to its saved standby pose without starting a repetition.
Press **Start** to begin the drill. After a repetition, press
**Replay/reset** to return to the saved start position; press **Start** again
for the next repetition. **End** exits the session and requests the configured
rest pose. Only `g`, `s`, `r`, and `q` are accepted by the listener. Keep it on
a trusted private network; the button remote is not an emergency stop.
