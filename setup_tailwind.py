import os
import subprocess
import sys

def run_command(command, description):
    print(f"--> {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"Success: {description}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error during '{description}':\n{e.stderr}")
        sys.exit(1)

def setup_tailwind():
    # 1. Initialize npm project if package.json does not exist
    if not os.path.exists("package.json"):
        run_command("npm init -y", "Initializing Node project (package.json)")

    # 2. Install Tailwind CSS CLI locally
    run_command("npm install -D tailwindcss", "Installing Tailwind CSS CLI")

    # 3. Create tailwind.config.js if missing
    if not os.path.exists("tailwind.config.js"):
        config_content = """module.exports = {
  content: ["./*.html", "./*.js", "./src/**/*.js"],
  theme: {
    extend: {},
  },
  plugins: [],
}"""
        with open("tailwind.config.js", "w", encoding="utf-8") as f:
            f.write(config_content)
        print("Created default tailwind.config.js")

    # 4. Create input CSS file
    os.makedirs("src", exist_ok=True)
    input_css_path = os.path.join("src", "input.css")
    if not os.path.exists(input_css_path):
        css_content = """@tailwind base;
@tailwind components;
@tailwind utilities;"""
        with open(input_css_path, "w", encoding="utf-8") as f:
            f.write(css_content)
        print("Created src/input.css")

    # 5. Build minified output CSS file
    os.makedirs("dist", exist_ok=True)
    run_command("npx tailwindcss -i ./src/input.css -o ./dist/output.css --minify", "Building compiled CSS bundle")

    print("\n✅ Tailwind CSS setup complete! File generated at ./dist/output.css")

if __name__ == "__main__":
    setup_tailwind()
