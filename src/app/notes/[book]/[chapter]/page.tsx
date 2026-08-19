import { getAllBooks, getNoteContent } from '@/lib/content';
import { MdxRenderer } from '@/components/md/MdxComponents';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { ArrowLeft, ChevronRight } from 'lucide-react';

export async function generateStaticParams() {
  const books = getAllBooks();
  const params: { book: string; chapter: string }[] = [];

  for (const book of books) {
    for (const chapter of book.chapters) {
      if (chapter.status === 'published') {
        params.push({ book: book.slug, chapter: chapter.slug });
      }
    }
  }

  return params;
}

export default function NotePage({ params }: { params: { book: string; chapter: string } }) {
  const books = getAllBooks();
  const bookMeta = books.find((b) => b.slug === params.book);
  const note = getNoteContent(params.book, params.chapter);

  if (!bookMeta || !note) {
    notFound();
  }

  // Chapter Navigation Logic
  const publishedChapters = bookMeta.chapters.filter((c) => c.status === 'published');
  const currentIndex = publishedChapters.findIndex((c) => c.slug === params.chapter);
  const prevChapter = publishedChapters[currentIndex - 1];
  const nextChapter = publishedChapters[currentIndex + 1];

  return (
    <article className="space-y-8 max-w-3xl mx-auto">
      {/* BREADCRUMB */}
      <nav className="flex items-center gap-2 text-sm text-neutral-500 font-mono">
        <Link href="/notes" className="hover:underline">Notes</Link>
        <ChevronRight size={14} />
        <Link href={`/notes#${bookMeta.slug}`} className="hover:underline truncate max-w-[200px]">{bookMeta.title}</Link>
      </nav>

      <header className="border-b pb-6">
        <h1 className="text-3xl font-bold tracking-tight mb-2">{note.title}</h1>
        <p className="text-sm text-neutral-500">From book: <span className="italic">{bookMeta.title}</span> by {bookMeta.author}</p>
      </header>

      <MdxRenderer source={note.content} />

      {/* NEXT / PREVIOUS CHAPTER NAV */}
      <footer className="border-t pt-6 mt-12 grid grid-cols-2 gap-4">
        {prevChapter ? (
          <Link href={`/notes/${params.book}/${prevChapter.slug}`} className="p-4 border rounded-lg hover:border-neutral-400">
            <span className="text-xs text-neutral-500 block mb-1">Previous</span>
            <span className="text-sm font-medium">{prevChapter.title}</span>
          </Link>
        ) : <div />}

        {nextChapter ? (
          <Link href={`/notes/${params.book}/${nextChapter.slug}`} className="p-4 border rounded-lg hover:border-neutral-400 text-right">
            <span className="text-xs text-neutral-500 block mb-1">Next</span>
            <span className="text-sm font-medium">{nextChapter.title}</span>
          </Link>
        ) : <div />}
      </footer>
    </article>
  );
}
