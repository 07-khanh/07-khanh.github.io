import './globals.css';
import 'katex/dist/katex.min.css';
import 'highlight.js/styles/github-dark.css';
import { Inter, Newsreader, JetBrains_Mono } from 'next/font/google';
import { Navigation } from '@/components/Navigation';

const inter = Inter({ subsets: ['latin'], variable: '--font-sans' });
const newsreader = Newsreader({ subsets: ['latin'], variable: '--font-serif' });
const mono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-mono' });

export const metadata = {
  title: 'Personal Technical Library',
  description: 'Blog, study notes, and projects.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${newsreader.variable} ${mono.variable}`}>
      <body className="font-serif bg-[#fafaf9] text-[#262626] min-h-screen flex flex-col">
        <Navigation />
        <main className="flex-1 max-w-3xl mx-auto px-6 py-12 w-full">
          {children}
        </main>
        <footer className="max-w-3xl mx-auto px-6 py-12 w-full text-center text-sm text-gray-500 font-sans mt-12 border-t border-gray-200">
          © {new Date().getFullYear()} Technical Library.
        </footer>
      </body>
    </html>
  );
}
