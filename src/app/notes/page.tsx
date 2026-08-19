import { getBooks } from '@/lib/content';
import Link from 'next/link';

export default function NotesIndex() {
  const books = getBooks();
  return (
    <div className="space-y-12">
      <header className="border-b border-gray-200 pb-8">
        <h1 className="text-3xl font-sans font-bold text-gray-900">Study Notes</h1>
        <p className="text-gray-600 mt-2 font-serif">Detailed technical notes from my reading.</p>
      </header>
      <div className="grid gap-6">
        {books.map(book => (
          <Link href={`/notes/${book?.slug}`} key={book?.slug} className="block p-8 bg-white border border-gray-200 rounded-2xl shadow-sm hover:border-amber-400 hover:shadow-md transition-all">
            <h2 className="text-2xl font-sans font-bold text-gray-900">{book?.title}</h2>
            <p className="text-sm font-sans font-medium text-gray-500 mt-2 uppercase tracking-wide">By {book?.author}</p>
            <p className="text-gray-600 mt-4 font-serif leading-relaxed">{book?.description}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
