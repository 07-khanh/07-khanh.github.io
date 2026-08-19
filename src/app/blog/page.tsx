import { getPosts } from '@/lib/content';
import Link from 'next/link';

export default function BlogList() {
  const posts = getPosts();
  return (
    <div className="space-y-12">
      <header className="border-b border-gray-200 pb-8">
        <h1 className="text-3xl font-sans font-bold text-gray-900">Blog</h1>
        <p className="text-gray-600 mt-2 font-serif">Thoughts, tutorials, and articles.</p>
      </header>
      <div className="space-y-10">
        {posts.map(post => (
          <article key={post.slug} className="group">
            <Link href={`/blog/${post.slug}`} className="block">
              <time className="text-sm font-sans text-gray-400 block mb-2">{post.date}</time>
              <h2 className="text-xl font-sans font-bold text-gray-900 group-hover:text-amber-600 transition-colors">
                {post.title}
              </h2>
              <p className="text-gray-600 mt-2 font-serif leading-relaxed">{post.description}</p>
            </Link>
          </article>
        ))}
      </div>
    </div>
  );
}
