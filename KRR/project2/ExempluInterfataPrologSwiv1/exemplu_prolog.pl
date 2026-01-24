:-use_module(library(socket)).


inceput:-format('Salutare\n',[]),	flush_output,
				current_prolog_flag(argv, [PortSocket|_]), %preiau numarul portului, dat ca argument cu -a
				%portul este atom, nu constanta numerica, asa ca trebuie sa il convertim la numar
				write('port '),
				write(PortSocket),nl,
				atom_chars(PortSocket,LCifre),
				number_chars(Port,LCifre),%transforma lista de cifre in numarul din 
				tcp_socket(Socket),
				tcp_connect(Socket, localhost:Port),
				tcp_open_socket(Socket,IStream, OStream),
				proceseaza_text_primit(IStream, OStream,0).
							
				
proceseaza_text_primit(IStream, OStream,C):-
				read(IStream,CevaCitit),
				write('Am primit':CevaCitit),nl,
				proceseaza_termen_citit(IStream, OStream,CevaCitit,C).
				
proceseaza_termen_citit(IStream, OStream,salut,C):-
				write(OStream,'Salut, de la SWI!\n'),
				flush_output(OStream),
				C1 is C+1,
				proceseaza_text_primit(IStream, OStream,C1).


proceseaza_termen_citit(IStream, OStream,'ce mai faci?',C):-
				write(IStream, OStream,'ma plictisesc...\n'),
				flush_output(OStream),
				C1 is C+1,
				proceseaza_text_primit(IStream, OStream,C1).
				
proceseaza_termen_citit(IStream, OStream, X + Y,C):-
				Rez is X+Y,
				write(OStream,'SWI spune':Rez),nl(OStream),
				flush_output(OStream),
				C1 is C+1,
				proceseaza_text_primit(IStream, OStream,C1).
				
oras(bucuresti, mare).
oras(constanta, mare).				
oras(pitesti, mediu).	
oras(buftea, mic).
proceseaza_termen_citit(IStream, OStream, oras(X),C):-
				oras(X,Tip),
				format(OStream,'~p este un oras ~p\n',[X,Tip]),
				flush_output(OStream),
				C1 is C+1,
				proceseaza_text_primit(IStream, OStream,C1).

proceseaza_termen_citit(IStream, OStream, X, _):-
				(X == end_of_file ; X == exit),
				close(IStream),
				close(OStream).
				
			
proceseaza_termen_citit(IStream, OStream, Altceva,C):-
				write(OStream,'nu inteleg ce vrei sa spui: '),write(OStream,Altceva),nl(OStream),
				flush_output(OStream),
				C1 is C+1,
				proceseaza_text_primit(IStream, OStream,C1).