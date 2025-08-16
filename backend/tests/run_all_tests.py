"""
Comprehensive Test Runner for Industry Reporter 2
Runs all end-to-end tests and generates a summary report
"""
import asyncio
import sys
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test modules
from test_redis_service import run_redis_tests
from test_faiss_service import run_faiss_tests
from test_document_loader import run_document_loader_tests
from test_multi_retriever_system import run_multi_retriever_tests
from test_skills_modules import run_skills_tests


class TestRunner:
    """Comprehensive test runner for all modules"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
        # Define test suites
        self.test_suites = [
            {
                'name': 'Redis Service',
                'function': run_redis_tests,
                'description': 'Tests Redis caching, serialization, namespaces, and performance',
                'critical': True
            },
            {
                'name': 'FAISS Service', 
                'function': run_faiss_tests,
                'description': 'Tests vector search, document indexing, similarity search, and MMR',
                'critical': True
            },
            {
                'name': 'Document Loader',
                'function': run_document_loader_tests,
                'description': 'Tests document parsing, format support, metadata extraction',
                'critical': True
            },
            {
                'name': 'Multi-Retriever System',
                'function': run_multi_retriever_tests,
                'description': 'Tests retriever integration, parallel search, result merging',
                'critical': True
            },
            {
                'name': 'Skills Modules',
                'function': run_skills_tests,
                'description': 'Tests ContextManager, ResearchConductor, ReportGenerator integration',
                'critical': True
            }
        ]
    
    async def run_all_tests(self, verbose: bool = True, stop_on_failure: bool = False) -> Dict:
        """Run all test suites and return comprehensive results"""
        
        self.start_time = datetime.now()
        
        if verbose:
            print("🚀 Industry Reporter 2 - Comprehensive End-to-End Testing")
            print("=" * 70)
            print(f"Started at: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Running {len(self.test_suites)} test suites...\n")
        
        total_passed = 0
        total_failed = 0
        
        for i, test_suite in enumerate(self.test_suites, 1):
            suite_name = test_suite['name']
            
            if verbose:
                print(f"📋 [{i}/{len(self.test_suites)}] Running {suite_name} Tests")
                print(f"    {test_suite['description']}")
                print("-" * 50)
            
            suite_start = time.time()
            
            try:
                # Run the test suite
                result = await test_suite['function']()
                suite_end = time.time()
                duration = suite_end - suite_start
                
                if result:
                    status = "PASSED"
                    total_passed += 1
                    if verbose:
                        print(f"✅ {suite_name} tests PASSED ({duration:.2f}s)")
                else:
                    status = "FAILED"
                    total_failed += 1
                    if verbose:
                        print(f"❌ {suite_name} tests FAILED ({duration:.2f}s)")
                    
                    if stop_on_failure and test_suite['critical']:
                        if verbose:
                            print(f"🛑 Stopping on critical failure: {suite_name}")
                        break
                
                self.test_results[suite_name] = {
                    'status': status,
                    'passed': result,
                    'duration': duration,
                    'critical': test_suite['critical'],
                    'description': test_suite['description']
                }
                
            except Exception as e:
                suite_end = time.time()
                duration = suite_end - suite_start
                
                if verbose:
                    print(f"💥 {suite_name} tests CRASHED: {str(e)} ({duration:.2f}s)")
                
                self.test_results[suite_name] = {
                    'status': 'CRASHED',
                    'passed': False,
                    'duration': duration,
                    'critical': test_suite['critical'],
                    'description': test_suite['description'],
                    'error': str(e)
                }
                
                total_failed += 1
                
                if stop_on_failure and test_suite['critical']:
                    if verbose:
                        print(f"🛑 Stopping on critical crash: {suite_name}")
                    break
            
            if verbose:
                print()  # Add spacing between test suites
        
        self.end_time = datetime.now()
        
        # Generate summary
        summary = self._generate_summary(total_passed, total_failed, verbose)
        
        return summary
    
    def _generate_summary(self, total_passed: int, total_failed: int, verbose: bool = True) -> Dict:
        """Generate comprehensive test summary"""
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        total_tests = total_passed + total_failed
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_duration': total_duration,
            'total_suites': len(self.test_suites),
            'suites_passed': total_passed,
            'suites_failed': total_failed,
            'success_rate': success_rate,
            'overall_status': 'PASSED' if total_failed == 0 else 'FAILED',
            'test_results': self.test_results
        }
        
        if verbose:
            self._print_summary(summary)
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print detailed test summary"""
        
        print("📊 Test Execution Summary")
        print("=" * 70)
        print(f"Start Time:      {summary['start_time']}")
        print(f"End Time:        {summary['end_time']}")
        print(f"Total Duration:  {summary['total_duration']:.2f} seconds")
        print(f"Success Rate:    {summary['success_rate']:.1f}%")
        print()
        
        # Print individual test results
        print("📋 Individual Test Suite Results:")
        print("-" * 70)
        
        for suite_name, result in summary['test_results'].items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌" if result['status'] == 'FAILED' else "💥"
            critical_mark = " [CRITICAL]" if result['critical'] else ""
            
            print(f"{status_icon} {suite_name}{critical_mark}")
            print(f"    Status: {result['status']}")
            print(f"    Duration: {result['duration']:.2f}s")
            print(f"    Description: {result['description']}")
            
            if 'error' in result:
                print(f"    Error: {result['error']}")
            
            print()
        
        # Print overall verdict
        print("🏁 Overall Test Result")
        print("-" * 70)
        
        if summary['overall_status'] == 'PASSED':
            print("🎉 ALL TESTS PASSED! Industry Reporter 2 is ready for deployment.")
            print("   All core functionalities are working correctly:")
            print("   ✅ Redis caching and data persistence")
            print("   ✅ FAISS vector search and similarity matching")
            print("   ✅ Document loading and processing")
            print("   ✅ Multi-retriever search integration")
            print("   ✅ Skills modules and research workflow")
        else:
            print("⚠️  SOME TESTS FAILED! Please review and fix issues before deployment.")
            
            failed_critical = [
                name for name, result in summary['test_results'].items()
                if not result['passed'] and result['critical']
            ]
            
            if failed_critical:
                print("   🚨 Critical failures that must be fixed:")
                for suite_name in failed_critical:
                    print(f"      ❌ {suite_name}")
            
            failed_non_critical = [
                name for name, result in summary['test_results'].items()
                if not result['passed'] and not result['critical']
            ]
            
            if failed_non_critical:
                print("   ⚠️  Non-critical failures (recommended to fix):")
                for suite_name in failed_non_critical:
                    print(f"      ❌ {suite_name}")
        
        print()
    
    def save_results(self, filename: str = None):
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        import json
        
        with open(filename, 'w') as f:
            json.dump({
                'test_summary': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': self.end_time.isoformat(),
                    'total_duration': (self.end_time - self.start_time).total_seconds(),
                    'success_rate': (len([r for r in self.test_results.values() if r['passed']]) / 
                                   len(self.test_results) * 100) if self.test_results else 0
                },
                'test_results': self.test_results
            }, f, indent=2)
        
        print(f"📄 Test results saved to: {filename}")
    
    async def run_health_check(self) -> Dict:
        """Run a quick health check of all systems"""
        print("🔍 Running System Health Check...")
        print("-" * 40)
        
        health_results = {}
        
        # Check Redis connectivity
        try:
            from services.redis_service import RedisService
            redis_service = RedisService(redis_url="redis://localhost:6379/12")
            await redis_service.initialize()
            health_check = await redis_service.health_check()
            await redis_service.close()
            
            health_results['redis'] = {
                'status': health_check.get('status', 'unknown'),
                'latency_ms': health_check.get('latency_ms', 0)
            }
            
        except Exception as e:
            health_results['redis'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Check FAISS functionality
        try:
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            faiss_path = os.path.join(temp_dir, "health_check_faiss")
            
            from services.faiss_service import FAISSService
            faiss_service = FAISSService(
                index_path=faiss_path,
                dimension=1536,
                index_type='Flat'
            )
            await faiss_service.initialize()
            health_check = await faiss_service.health_check()
            
            shutil.rmtree(temp_dir)
            
            health_results['faiss'] = {
                'status': health_check.get('status', 'unknown'),
                'latency_ms': health_check.get('latency_ms', 0)
            }
            
        except Exception as e:
            health_results['faiss'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Print health check results
        for service, result in health_results.items():
            status_icon = "✅" if result['status'] == 'healthy' else "❌"
            print(f"{status_icon} {service.upper()}: {result['status']}")
            
            if 'latency_ms' in result:
                print(f"    Latency: {result['latency_ms']:.2f}ms")
            
            if 'error' in result:
                print(f"    Error: {result['error']}")
        
        print()
        return health_results


async def main():
    """Main function to run tests"""
    
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Industry Reporter 2 Test Runner')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--stop-on-failure', '-s', action='store_true',
                       help='Stop testing on first critical failure')
    parser.add_argument('--health-check-only', '-h', action='store_true',
                       help='Run only health check, skip full tests')
    parser.add_argument('--save-results', '-r', action='store_true',
                       help='Save test results to JSON file')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner()
    
    if args.health_check_only:
        # Run health check only
        health_results = await runner.run_health_check()
        return 0 if all(r.get('status') == 'healthy' for r in health_results.values()) else 1
    
    # Run full test suite
    summary = await runner.run_all_tests(
        verbose=args.verbose,
        stop_on_failure=args.stop_on_failure
    )
    
    # Save results if requested
    if args.save_results:
        runner.save_results()
    
    # Return exit code based on overall results
    return 0 if summary['overall_status'] == 'PASSED' else 1


if __name__ == "__main__":
    # Run tests
    exit_code = asyncio.run(main())
    sys.exit(exit_code)