env: env/bin/.install
env/bin/.install:
		python3.7 -m venv env
		env/bin/pip install -r ./requirements/requirements.txt
		touch $@
