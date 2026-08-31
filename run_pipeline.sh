#!/bin/bash

set -e  # Exit on any error

echo "============================================"
echo "🐝 Academic Bee Pipeline - Full Automation"
echo "============================================"
echo ""

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ ERROR: GEMINI_API_KEY environment variable is not set."
    echo ""
    echo "Please set your API key:"
    echo "  export GEMINI_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

echo "✓ GEMINI_API_KEY is configured"
echo ""

# Step 1: Install Dependencies
echo "📦 Step 1: Installing dependencies..."
python -m pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

# Step 2: Run AI Research Agent
echo "🤖 Step 2: Running AI Research Agent..."
echo "   - Scraping PDFs from IAC website..."
echo "   - Building vector database..."
echo "   - Generating 30 pyramidal tossup questions..."
python agent.py
echo "✓ Generated quizzes.json"
echo ""

# Step 3: Run Knowledge Base Builder
echo "📚 Step 3: Building Knowledge Base..."
echo "   - Extracting topics from questions..."
echo "   - Generating summaries & facts via Gemini..."
python build_topics.py
echo "✓ Generated topics.json"
echo ""

# Step 4: Commit and Push (if git repo)
if [ -d ".git" ]; then
    echo "💾 Step 4: Committing to Git..."
    
    git config --local user.email "action@github.com" 2>/dev/null || true
    git config --local user.name "GitHub Action" 2>/dev/null || true
    
    # Only commit if there are changes
    if git diff --quiet topics.json 2>/dev/null; then
        echo "   ℹ️  No changes to topics.json, skipping commit"
    else
        git add topics.json quizzes.json 2>/dev/null || git add quizzes.json topics.json 2>/dev/null || true
        git commit -m "Auto-update topics.json and quizzes.json [skip ci]" 2>/dev/null || echo "   ℹ️  Nothing new to commit"
        
        if git push origin main 2>/dev/null; then
            echo "✓ Pushed updates to main branch"
        else
            echo "   ⚠️  Could not push (you may need to authenticate)"
        fi
    fi
    echo ""
else
    echo "⚠️  Not a git repository - skipping commit/push"
    echo ""
fi

# Step 5: Local Server (Optional)
echo "============================================"
echo "✅ Pipeline Complete!"
echo "============================================"
echo ""
echo "📊 Generated Files:"
echo "   - quizzes.json  (30 questions)"
echo "   - topics.json   (knowledge base)"
echo ""
echo "🌐 View locally:"
echo "   python -m http.server 8000"
echo "   Then open: http://localhost:8000"
echo ""
echo "🚀 To deploy to GitHub Pages:"
echo "   git push origin main"
echo "   (The Jekyll workflow will run automatically)"
echo ""
