#!/bin/bash
echo "========================================="
echo "VERIFYING CMGAN-STYLE CHANGES"
echo "========================================="
echo ""

echo "1. Checking test.py..."
if grep -q "CMGAN-Style Testing Script" "My Model/scripts/test.py"; then
    echo "   ✓ test.py has been updated"
else
    echo "   ✗ test.py NOT updated"
fi

echo ""
echo "2. Checking configs.py..."
if grep -q "'compute_metrics'" "My Model/scripts/configs.py"; then
    echo "   ✓ configs.py has CMGAN-style test_conf"
else
    echo "   ✗ configs.py NOT updated"
fi

echo ""
echo "3. Checking models.py..."
if grep -q "def enhance_one_track" "My Model/scripts/utils/models.py"; then
    echo "   ✓ models.py has enhance_one_track method"
else
    echo "   ✗ models.py NOT updated"
fi

echo ""
echo "4. Checking for new documentation..."
if [ -f "My Model/scripts/CMGAN_TESTING_CHANGES.md" ]; then
    echo "   ✓ CMGAN_TESTING_CHANGES.md exists"
    echo "   File size: $(du -h 'My Model/scripts/CMGAN_TESTING_CHANGES.md' | cut -f1)"
else
    echo "   ✗ CMGAN_TESTING_CHANGES.md NOT found"
fi

echo ""
echo "5. Checking git branch..."
BRANCH=$(git branch --show-current)
echo "   Current branch: $BRANCH"

echo ""
echo "6. Checking latest commit..."
git log -1 --oneline

echo ""
echo "========================================="
echo "VERIFICATION COMPLETE"
echo "========================================="
