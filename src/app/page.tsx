import Link from 'next/link';

export default function Home() {
  return (
    <div className="space-y-16 mt-8">
      <header className="space-y-6">
        <h1 className="text-4xl font-sans font-bold tracking-tight text-gray-900 leading-tight">
          Hi, I'm a Researcher <br/>& Software Engineer.
        </h1>
        <p className="text-lg text-gray-600 leading-relaxed max-w-2xl font-serif">
          This is my digital library. I write about machine learning, software engineering, and mathematics. I also publish detailed notes from the books and papers I study.
        </p>
      </header>

      <section>
        <h2 className="text-xs font-sans font-bold uppercase tracking-widest text-gray-400 mb-6">Explore the Library</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Link href="/notes" className="block p-6 rounded-2xl bg-white border border-gray-200 shadow-sm hover:border-amber-400 transition-all group">
            <h3 className="font-sans font-semibold text-lg text-gray-900 group-hover:text-amber-600">Study Notes &rarr;</h3>
            <p className="text-gray-500 text-sm mt-2 font-serif leading-relaxed">Detailed technical notes, mathematical derivations, and code experiments.</p>
          </Link>
          <Link href="/blog" className="block p-6 rounded-2xl bg-white border border-gray-200 shadow-sm hover:border-amber-400 transition-all group">
            <h3 className="font-sans font-semibold text-lg text-gray-900 group-hover:text-amber-600">Blog &rarr;</h3>
            <p className="text-gray-500 text-sm mt-2 font-serif leading-relaxed">Tutorials, thoughts, and articles on technology and learning.</p>
          </Link>
        </div>
      </section>
    </div>
  );
}
