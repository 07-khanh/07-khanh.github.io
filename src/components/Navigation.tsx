import Link from 'next/link';

export function Navigation() {
  return (
    <nav className="max-w-3xl mx-auto px-6 py-8 flex justify-between items-center font-sans w-full">
      <Link href="/" className="font-bold text-lg tracking-tight text-gray-900 hover:text-amber-600 transition-colors">
        Library.
      </Link>
      <div className="flex gap-6 text-sm font-medium text-gray-600">
        <Link href="/blog" className="hover:text-gray-900 transition-colors">Blog</Link>
        <Link href="/notes" className="hover:text-gray-900 transition-colors">Notes</Link>
        <Link href="/projects" className="hover:text-gray-900 transition-colors">Projects</Link>
      </div>
    </nav>
  );
}
