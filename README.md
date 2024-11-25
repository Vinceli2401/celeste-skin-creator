# celeste-skin-creator
## Running
Run CelesteSkinCreator.py to generate a new skin.

Edit configurations in config.json.
## Packaging
Use this command:
````
pyinstaller --onefile --add-data "config.json:." --add-data "SkinBase:SkinBase" CelesteSkinCreator.py
