#!/usr/bin/env python3
"""
Migration script to upgrade from classification to segmentation
"""

import os
import shutil
import json
from datetime import datetime

def migrate_project():
    print("🔧 Migrating Mycorrhizal Detection System to Segmentation")
    print("=" * 55)
    
    # 1. Fix filename issue
    if os.path.exists("src/_init_.py"):
        print("📝 Fixing src/__init__.py filename...")
        os.rename("src/_init_.py", "src/__init__.py")
    
    # 2. Create new directories
    print("📁 Creating segmentation directories...")
    directories = [
        "data/segmentation/images",
        "data/segmentation/masks", 
        "data/segmentation/metadata",
        "models/segmentation",
        "src/segmentation"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}")
    
    # 3. Create segmentation package
    init_file = "src/segmentation/__init__.py"
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Mycorrhizal Segmentation Package\n")
        print("   ✅ src/segmentation/__init__.py")
    
    # 4. Backup original app
    if os.path.exists("app.py") and not os.path.exists("app_classification.py"):
        shutil.copy("app.py", "app_classification.py")
        print("   ✅ Backed up original app as app_classification.py")
    
    # 5. Create migration log
    migration_log = {
        "migration_date": datetime.now().isoformat(),
        "from_version": "classification",
        "to_version": "segmentation",
        "status": "completed"
    }
    
    with open("migration_log.json", 'w') as f:
        json.dump(migration_log, f, indent=2)
    
    print("\n✅ Migration completed successfully!")
    print("\n📋 Next steps:")
    print("1. Install new requirements: pip install -r requirements.txt")
    print("2. Test segmentation modules: python -c 'from src.segmentation.color_config import STRUCTURE_COLORS; print(\"✅ Segmentation ready\")'")
    print("3. Launch new app: streamlit run app_segmentation.py")

if __name__ == "__main__":
    migrate_project()
