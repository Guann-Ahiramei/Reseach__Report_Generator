"""
Document Loader End-to-End Tests
Comprehensive testing for document loading functionality
"""
import asyncio
import pytest
import tempfile
import shutil
import os
import json
import csv
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.document_loader import DocumentLoader, load_documents, load_single_document


class TestDocumentLoader:
    """Test document loader end-to-end functionality"""
    
    @pytest.fixture
    async def test_documents_dir(self):
        """Create temporary directory with test documents"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create test documents
            await self._create_test_documents(temp_dir)
            yield temp_dir
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
    
    async def _create_test_documents(self, base_dir: str):
        """Create various test documents"""
        # Text file
        with open(os.path.join(base_dir, "test_document.txt"), "w", encoding="utf-8") as f:
            f.write("""This is a test document for Industry Reporter 2.
It contains multiple lines of text content.
The document discusses artificial intelligence and machine learning applications.
Industry analysis shows significant growth in AI adoption.
Companies are investing heavily in AI-powered solutions.""")
        
        # Markdown file
        with open(os.path.join(base_dir, "readme.md"), "w", encoding="utf-8") as f:
            f.write("""# Industry Reporter Documentation
            
## Overview
This document provides an overview of the Industry Reporter system.

### Features
- Multi-retriever search
- FAISS vector search  
- Redis caching
- Real-time WebSocket communication

### Getting Started
1. Install dependencies
2. Configure environment
3. Run the application
""")
        
        # JSON file
        test_data = {
            "report_id": "INDUSTRY_2024_001",
            "title": "AI Industry Trends Report",
            "sections": [
                {
                    "name": "Executive Summary",
                    "content": "The AI industry continues to show remarkable growth..."
                },
                {
                    "name": "Market Analysis", 
                    "content": "Key market segments include machine learning platforms..."
                }
            ],
            "metadata": {
                "created": datetime.now().isoformat(),
                "author": "Industry Research Team",
                "category": "Technology"
            }
        }
        
        with open(os.path.join(base_dir, "report.json"), "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        # CSV file
        csv_data = [
            ["Company", "Industry", "AI_Investment", "Year"],
            ["Google", "Technology", "15000000", "2024"],
            ["Microsoft", "Technology", "12000000", "2024"],
            ["Amazon", "E-commerce", "8000000", "2024"],
            ["Tesla", "Automotive", "5000000", "2024"]
        ]
        
        with open(os.path.join(base_dir, "companies.csv"), "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        
        # HTML file
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Industry Analysis</title>
</head>
<body>
    <h1>Industry Analysis Report</h1>
    <p>This report analyzes current trends in the technology industry.</p>
    <h2>Key Findings</h2>
    <ul>
        <li>AI adoption is accelerating across sectors</li>
        <li>Cloud computing revenue continues to grow</li>
        <li>Cybersecurity investments are increasing</li>
    </ul>
    <script>
        // This script should be removed during text extraction
        console.log("Script content");
    </script>
</body>
</html>"""
        
        with open(os.path.join(base_dir, "analysis.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Python code file
        python_code = '''"""
Industry data processing module
"""
import pandas as pd
import numpy as np

class IndustryAnalyzer:
    """Analyze industry trends and patterns"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
    
    def load_data(self):
        """Load industry data from CSV"""
        self.data = pd.read_csv(self.data_path)
        return self.data
    
    def analyze_trends(self):
        """Analyze industry trends"""
        if self.data is None:
            self.load_data()
        
        # Calculate growth rates
        growth_rates = self.data['AI_Investment'].pct_change()
        return growth_rates.describe()
'''
        
        with open(os.path.join(base_dir, "analyzer.py"), "w", encoding="utf-8") as f:
            f.write(python_code)
        
        # Create subdirectory with more files
        subdir = os.path.join(base_dir, "reports")
        os.makedirs(subdir)
        
        with open(os.path.join(subdir, "quarterly_report.txt"), "w", encoding="utf-8") as f:
            f.write("Quarterly industry performance shows positive trends in AI and cloud computing sectors.")
        
        # Create file that should be ignored (hidden)
        with open(os.path.join(base_dir, ".hidden_file.txt"), "w", encoding="utf-8") as f:
            f.write("This is a hidden file that should be ignored by default.")
        
        # Create large file for size testing
        large_content = "Large file content. " * 10000  # About 200KB
        with open(os.path.join(base_dir, "large_document.txt"), "w", encoding="utf-8") as f:
            f.write(large_content)
    
    async def test_loader_initialization(self, test_documents_dir):
        """Test document loader initialization"""
        loader = DocumentLoader(
            base_path=test_documents_dir,
            max_file_size_mb=1,
            recursive=True,
            include_hidden=False
        )
        
        assert loader.base_path == test_documents_dir
        assert loader.max_file_size_mb == 1
        assert loader.recursive == True
        assert loader.include_hidden == False
        
        # Test supported formats
        formats = await loader.get_supported_formats()
        assert isinstance(formats, list)
        assert '.txt' in formats
        assert '.json' in formats
        assert '.csv' in formats
        
        print("✅ Loader initialization test passed")
    
    async def test_single_file_loading(self, test_documents_dir):
        """Test loading individual files"""
        loader = DocumentLoader(base_path=test_documents_dir)
        
        # Test text file
        txt_file = os.path.join(test_documents_dir, "test_document.txt")
        txt_doc = await loader.load_file(txt_file)
        
        assert txt_doc is not None
        assert isinstance(txt_doc, dict)
        assert "content" in txt_doc
        assert "file_name" in txt_doc
        assert "file_type" in txt_doc
        assert "file_size" in txt_doc
        assert txt_doc["file_type"] == ".txt"
        assert "artificial intelligence" in txt_doc["content"].lower()
        
        # Test JSON file
        json_file = os.path.join(test_documents_dir, "report.json")
        json_doc = await loader.load_file(json_file)
        
        assert json_doc is not None
        assert json_doc["file_type"] == ".json"
        assert "AI Industry Trends" in json_doc["content"]
        
        # Test CSV file
        csv_file = os.path.join(test_documents_dir, "companies.csv")
        csv_doc = await loader.load_file(csv_file)
        
        assert csv_doc is not None
        assert csv_doc["file_type"] == ".csv"
        assert "Google" in csv_doc["content"]
        assert "Company" in csv_doc["content"]  # Column header
        
        # Test HTML file
        html_file = os.path.join(test_documents_dir, "analysis.html")
        html_doc = await loader.load_file(html_file)
        
        assert html_doc is not None
        assert html_doc["file_type"] == ".html"
        assert "Industry Analysis Report" in html_doc["content"]
        assert "console.log" not in html_doc["content"]  # Script should be removed
        
        # Test Python file
        py_file = os.path.join(test_documents_dir, "analyzer.py")
        py_doc = await loader.load_file(py_file)
        
        assert py_doc is not None
        assert py_doc["file_type"] == ".py"
        assert "class IndustryAnalyzer" in py_doc["content"]
        
        print("✅ Single file loading test passed")
    
    async def test_batch_loading(self, test_documents_dir):
        """Test loading all documents in directory"""
        loader = DocumentLoader(
            base_path=test_documents_dir,
            recursive=True,
            include_hidden=False
        )
        
        documents = await loader.load()
        
        assert isinstance(documents, list)
        assert len(documents) > 0
        
        # Check that we got various file types
        file_types = set(doc["file_type"] for doc in documents)
        expected_types = {".txt", ".json", ".csv", ".html", ".py", ".md"}
        
        # Should have found most expected types
        assert len(file_types.intersection(expected_types)) >= 4
        
        # Verify documents have required fields
        for doc in documents:
            assert "content" in doc
            assert "file_name" in doc
            assert "file_type" in doc
            assert "file_size" in doc
            assert "last_modified" in doc
            assert len(doc["content"]) > 0
        
        # Check that hidden files were excluded
        hidden_files = [doc for doc in documents if doc["file_name"].startswith(".")]
        assert len(hidden_files) == 0
        
        print("✅ Batch loading test passed")
    
    async def test_recursive_loading(self, test_documents_dir):
        """Test recursive directory loading"""
        loader = DocumentLoader(
            base_path=test_documents_dir,
            recursive=True
        )
        
        documents = await loader.load()
        
        # Should find files in subdirectories
        subdir_files = [doc for doc in documents if "quarterly_report" in doc["file_name"]]
        assert len(subdir_files) > 0
        
        # Test non-recursive
        loader_non_recursive = DocumentLoader(
            base_path=test_documents_dir,
            recursive=False
        )
        
        non_recursive_docs = await loader_non_recursive.load()
        recursive_docs = documents
        
        # Recursive should find more files
        assert len(recursive_docs) >= len(non_recursive_docs)
        
        print("✅ Recursive loading test passed")
    
    async def test_file_size_filtering(self, test_documents_dir):
        """Test file size filtering"""
        # Test with very small size limit
        small_loader = DocumentLoader(
            base_path=test_documents_dir,
            max_file_size_mb=0.1  # 100KB limit
        )
        
        small_docs = await small_loader.load()
        
        # Test with larger size limit
        large_loader = DocumentLoader(
            base_path=test_documents_dir,
            max_file_size_mb=1  # 1MB limit
        )
        
        large_docs = await large_loader.load()
        
        # Should find more files with larger limit
        assert len(large_docs) >= len(small_docs)
        
        # Check file sizes
        for doc in small_docs:
            assert doc["file_size"] <= 0.1 * 1024 * 1024  # 100KB in bytes
        
        print("✅ File size filtering test passed")
    
    async def test_metadata_extraction(self, test_documents_dir):
        """Test metadata extraction"""
        loader = DocumentLoader(
            base_path=test_documents_dir,
            extract_metadata=True
        )
        
        documents = await loader.load()
        
        # Check that metadata is included
        for doc in documents:
            assert "metadata" in doc
            metadata = doc["metadata"]
            
            assert "file_path" in metadata
            assert "file_name" in metadata
            assert "file_type" in metadata
            assert "file_size" in metadata
            assert "content_length" in metadata
            assert "word_count" in metadata
            assert "line_count" in metadata
            
            # Verify metadata values make sense
            assert metadata["content_length"] == len(doc["content"])
            assert metadata["word_count"] > 0 if doc["content"] else True
            assert metadata["line_count"] >= 1 if doc["content"] else True
        
        print("✅ Metadata extraction test passed")
    
    async def test_content_filtering(self, test_documents_dir):
        """Test content length filtering"""
        loader = DocumentLoader(
            base_path=test_documents_dir,
            min_content_length=50  # Minimum 50 characters
        )
        
        documents = await loader.load()
        
        # All documents should meet minimum length
        for doc in documents:
            assert len(doc["content"]) >= 50
        
        # Test stats
        stats = loader.get_stats()
        assert isinstance(stats, dict)
        assert "files_processed" in stats
        assert "files_skipped" in stats
        assert "total_size_bytes" in stats
        
        print("✅ Content filtering test passed")
    
    async def test_encoding_detection(self, test_documents_dir):
        """Test encoding detection and handling"""
        # Create file with different encoding
        utf8_file = os.path.join(test_documents_dir, "utf8_test.txt")
        with open(utf8_file, "w", encoding="utf-8") as f:
            f.write("Test with UTF-8 encoding: 中文测试 🚀")
        
        loader = DocumentLoader(
            base_path=test_documents_dir,
            detect_encoding=True
        )
        
        doc = await loader.load_file(utf8_file)
        
        assert doc is not None
        assert "中文测试" in doc["content"]
        assert "🚀" in doc["content"]
        assert doc["encoding"] in ["utf-8", "UTF-8"]
        
        print("✅ Encoding detection test passed")
    
    async def test_file_support_checking(self, test_documents_dir):
        """Test file support checking"""
        loader = DocumentLoader(base_path=test_documents_dir)
        
        # Test supported file
        txt_file = os.path.join(test_documents_dir, "test_document.txt")
        support_result = await loader.test_file_support(txt_file)
        
        assert isinstance(support_result, dict)
        assert support_result["supported"] == True
        assert "file_info" in support_result
        assert "content_preview" in support_result
        
        # Test unsupported file (create one)
        unsupported_file = os.path.join(test_documents_dir, "test.unknown")
        with open(unsupported_file, "w") as f:
            f.write("Unsupported format")
        
        unsupported_result = await loader.test_file_support(unsupported_file)
        assert unsupported_result["supported"] == False
        assert "reason" in unsupported_result
        
        # Test non-existent file
        nonexistent_result = await loader.test_file_support("nonexistent.txt")
        assert nonexistent_result["supported"] == False
        assert "File does not exist" in nonexistent_result["reason"]
        
        print("✅ File support checking test passed")
    
    async def test_error_handling(self, test_documents_dir):
        """Test error handling and edge cases"""
        loader = DocumentLoader(base_path=test_documents_dir)
        
        # Test loading non-existent file
        result = await loader.load_file("nonexistent.txt")
        assert result is None
        
        # Test loading from non-existent directory
        bad_loader = DocumentLoader(base_path="/nonexistent/directory")
        documents = await bad_loader.load()
        assert isinstance(documents, list)
        assert len(documents) == 0
        
        # Test with empty file
        empty_file = os.path.join(test_documents_dir, "empty.txt")
        with open(empty_file, "w") as f:
            pass  # Create empty file
        
        empty_doc = await loader.load_file(empty_file)
        # Should be None or skipped due to min content length
        
        print("✅ Error handling test passed")
    
    async def test_convenience_functions(self, test_documents_dir):
        """Test convenience functions"""
        # Test load_documents function
        documents = await load_documents(test_documents_dir, recursive=True)
        assert isinstance(documents, list)
        assert len(documents) > 0
        
        # Test load_single_document function
        txt_file = os.path.join(test_documents_dir, "test_document.txt")
        document = await load_single_document(txt_file)
        assert document is not None
        assert document["file_type"] == ".txt"
        
        print("✅ Convenience functions test passed")


async def run_document_loader_tests():
    """Run all document loader tests"""
    print("🚀 Starting Document Loader End-to-End Tests\n")
    
    try:
        # Create test instance
        test_instance = TestDocumentLoader()
        
        # Create temporary directory with test documents
        temp_dir = tempfile.mkdtemp()
        await test_instance._create_test_documents(temp_dir)
        
        print("📋 Running document loader tests...")
        
        # Run all tests
        await test_instance.test_loader_initialization(temp_dir)
        await test_instance.test_single_file_loading(temp_dir)
        await test_instance.test_batch_loading(temp_dir)
        await test_instance.test_recursive_loading(temp_dir)
        await test_instance.test_file_size_filtering(temp_dir)
        await test_instance.test_metadata_extraction(temp_dir)
        await test_instance.test_content_filtering(temp_dir)
        await test_instance.test_encoding_detection(temp_dir)
        await test_instance.test_file_support_checking(temp_dir)
        await test_instance.test_error_handling(temp_dir)
        await test_instance.test_convenience_functions(temp_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n🎉 All document loader tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Document loader tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    result = asyncio.run(run_document_loader_tests())
    exit(0 if result else 1)