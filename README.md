# Underwater Color Correction using ML

This README is under construction, but the project consists of using GANs to generate a synthetic dataset to train a CNN to correct the color of underwater images.

## Python environment

If it is your first time running the project, go from step 1 to 5.
If you already installed the requeriments and want to run it, just activate the environment with the step 3.

1. Install python3-venv:

```bash
sudo apt-get install python3-venv -y
```

2. In the root path of this repo, run:

```bash
python3 -m venv v-env
```

3. Activate the environment:

```bash
source v-env/bin/activate
```

4. Run this before requirements so it doesn't fail:

```bash
pip install wheel
```

5. Install the project requirements:

```bash
pip install -r requirements.txt
```

## Running the GAN

1. Make sure your environment is activated:

```bash
source v-env/bin/activate
```

2. To train the GAN, run:

```bash
cd gan/
python main.py
```

## Authors and Contributors

- Gabriel Noya - [gnoya](https://github.com/gnoya 'https://github.com/gnoya')
- Jose Cappelletto - [cappelletto](https://github.com/cappelletto 'https://github.com/cappelletto')
