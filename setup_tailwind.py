import subprocess
import sys

def install_and_build_tailwind():
    print("Installing Tailwind CSS via npm...")
    # Install Tailwind CLI locally
    subprocess.run(["npm", "install", "-D", "tailwindcss"], check=True)
    
    # Initialize tailwind config if it doesn't exist
    subprocess.run(["npx", "tailwindcss", "init", "-p"], check=False)
    
    print("Building standalone CSS file...")
    # Generate minified CSS output
    subprocess.run([
        "npx", "tailwindcss", 
        "-i", "./src/input.css", 
        "-o", "./dist/output.css", 
        "--minify"
    ], check=True)
    print("Tailwind CSS build complete! Linked to ./dist/output.css")

if __name__ == "__main__":
    install_and_build_tailwind()
