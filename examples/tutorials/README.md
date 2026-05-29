# Tutorials

Collection of notebooks that walk users through various end-to-end workflows. The notebooks here provide workflows for developers to get started with data collection, physical AI model fine tuning and testing the OpenVINO™ Physical AI APIs for deploying the trained policies.


**Getting Started**
```bash
pip install jupyter # if jupyter notebook is not already installed
phython -m venv venv
source venv/bin/activate
git clone https://github.com/openvinotoolkit/physicalai.git
cd physicalai/examples/notebooks
jupyter lab
```

### List of notebooks:

| **Notebook** | **Description** |
|:-------------|:----------------|
| [001_Introduction](001_Introduction.ipynb) | Introduction to different pathways to test OpenVINO Physical AI |
| [002_Using_Physical_AI_Studio](002_Using_Physical_AI_Studio.ipynb) | Utilizing Physical AI Studio for full workflow with built-in OpenVINO Physical AI API, with a physical robot |
| [003_OpenVINO_Optimization](003_OpenVINO_Optimization.ipynb) | Bring model from Physical AI Studio or Lerobot, optimize with OpenVINO, and deploy using OpenVINO Physical AI API |
| [004_Test_Deployment_Without_Robot](004_Test_Deployment_Without_Robot.ipynb) | Test deployment on a subset of dataset, without a physical robot |
| [005_Collect_Train_Deploy_SO101](005_collect_train_deploy.ipynb) | Workflow for data collection, model fine tuning, and deployment with SO101 arms and π0.5 visuomotor diffusion policy |


### Related documentation

- [Run a Policy on a Robot](../../docs/how-to/runtime/run-policy-on-robot.md)
- [Load an Exported Policy](../../docs/how-to/inference/load-exported-policy.md)
- [Robot API reference](../../docs/reference/robot-api.md)
