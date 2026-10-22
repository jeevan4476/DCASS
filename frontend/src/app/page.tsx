import Link from 'next/link';
import Navigation from '@/components/Navigation';

export default function Home() {
  return (
    <>
      <Navigation />
      <main className="min-h-screen bg-background">
        <div className="container mx-auto px-4 py-16">
          <div className="max-w-4xl mx-auto text-center">
            <h1 className="text-5xl font-bold text-primary mb-6">
              DCASS
            </h1>
            <p className="text-2xl text-gray-400 mb-8">
              Dynamic Context-Aware Semantic Steganography
            </p>
            <p className="text-lg text-gray-300 mb-12 max-w-2xl mx-auto">
              Zero-modification semantic steganography system with AI-driven stealth.
              Encode messages using semantically aligned media without altering any carrier content.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-12">
              <Link 
                href="/status"
                className="bg-gray-900 border border-gray-800 rounded-lg p-6 hover:border-primary transition-colors"
              >
                <div className="text-3xl mb-3">📊</div>
                <h3 className="text-lg font-semibold text-white mb-2">System Status</h3>
                <p className="text-sm text-gray-400">
                  View corpus statistics, model status, and system health
                </p>
              </Link>

              <Link 
                href="/encode"
                className="bg-gray-900 border border-gray-800 rounded-lg p-6 hover:border-primary transition-colors"
              >
                <div className="text-3xl mb-3">🔐</div>
                <h3 className="text-lg font-semibold text-white mb-2">Encode Message</h3>
                <p className="text-sm text-gray-400">
                  Transform secret messages into semantic media sequences
                </p>
              </Link>

              <Link 
                href="/decode"
                className="bg-gray-900 border border-gray-800 rounded-lg p-6 hover:border-primary transition-colors"
              >
                <div className="text-3xl mb-3">🔎</div>
                <h3 className="text-lg font-semibold text-white mb-2">Decode Sequence</h3>
                <p className="text-sm text-gray-400">
                  Reconstruct semantic meaning from media IDs using loaded indices
                </p>
              </Link>

              <Link 
                href="/wire"
                className="bg-gray-900 border border-gray-800 rounded-lg p-6 hover:border-primary transition-colors"
              >
                <div className="text-3xl mb-3">📡</div>
                <h3 className="text-lg font-semibold text-white mb-2">Wire View</h3>
                <p className="text-sm text-gray-400">
                  Real-time transmission telemetry and packet monitoring
                </p>
              </Link>
            </div>

            <div className="bg-gray-900 border border-gray-800 rounded-lg p-8 text-left">
              <h2 className="text-xl font-semibold text-primary mb-4">Key Features</h2>
              <ul className="space-y-3 text-gray-300">
                <li className="flex items-start">
                  <span className="text-success mr-3">✓</span>
                  <span><strong>Zero-Modification:</strong> No changes to carrier media — resistant to classical steganalysis</span>
                </li>
                <li className="flex items-start">
                  <span className="text-success mr-3">✓</span>
                  <span><strong>Multi-Modal:</strong> Support for text, image, and audio semantic encoding</span>
                </li>
                <li className="flex items-start">
                  <span className="text-success mr-3">✓</span>
                  <span><strong>Dynamic Fallback:</strong> Auto mode tries RL → GAN → Static scheduling</span>
                </li>
                <li className="flex items-start">
                  <span className="text-success mr-3">✓</span>
                  <span><strong>Behavioral Stealth:</strong> Human-like transmission patterns with noise injection</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
