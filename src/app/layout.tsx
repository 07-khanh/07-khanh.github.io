import type { Metadata } from 'next';
import Link from 'next/link';
import './globals.css';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github-dark.css';
import { Inter, Newsreader, JetBrains_Mono } from 'next/font/google';
import { Navigation } from '@/components/Navigation';
import { site } from '@/lib/site';

const inter = Inter({ subsets: ['latin', 'vietnamese'], variable: '--font-sans', display: 'swap' });
const newsreader = Newsreader({ subsets: ['latin', 'vietnamese'], variable: '--font-serif', display: 'swap', adjustFontFallback: false, fallback: ['Georgia', 'serif'] });
const mono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono', display: 'swap' });

export const metadata: Metadata = {
  metadataBase: new URL(site.url),
  title: { default: `${site.name} · AI & ML`, template: `%s · ${site.name}` },
  description: site.description,
  openGraph: { type: 'website', siteName: site.name, locale: 'en_US', title: site.name, description: site.description },
  twitter: { card: 'summary', title: site.name, description: site.description },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${newsreader.variable} ${mono.variable}`}>
      <body>
        <a href="#main-content" className="skip-link">Skip to content</a>
        <Navigation />
        <main id="main-content" tabIndex={-1} className="shell main-content">{children}</main>
        <footer className="shell site-footer">
          <div><Link href="/">{site.name}</Link><p>Learning, building, and documenting along the way.</p></div>
          <a href={site.github}>GitHub <span aria-hidden="true">↗</span></a>
        </footer>
      </body>
    </html>
  );
}
